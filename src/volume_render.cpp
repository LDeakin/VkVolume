/* Copyright (c) 2019, Lachlan Deakin
*
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 the "License";
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "volume_render.h"

#include "benchmark_mode/benchmark_mode.h"
#include "common/vk_initializers.h"
#include "glsl_compiler.h"
#include "gltf_loader.h"
#include "gui.h"
#include "platform/filesystem.h"
#include "platform/parser.h"
#include "platform/parsers/CLI11.h"
#include "platform/platform.h"
#include "rendering/render_context.h"
#include "rendering/render_pipeline.h"
#include "rendering/subpasses/forward_subpass.h"
#include "scene_graph/components/perspective_camera.h"

#if defined(VK_USE_PLATFORM_ANDROID_KHR)
#	include "platform/android/android_platform.h"
#endif

VKBP_DISABLE_WARNINGS()
#include <glm/gtx/matrix_decompose.hpp>
#include <spdlog/sinks/basic_file_sink.h>
#if defined(VK_USE_PLATFORM_WIN32_KHR)
#	include <spdlog/sinks/msvc_sink.h>
#endif
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
VKBP_ENABLE_WARNINGS()

#include "volume_render_subpass.h"

using namespace vkb;

VolumeRenderPlugin::VolumeRenderPlugin() :
    VolumeRenderPluginTags("VolumeRender",
                           "VolumeRender input options",
                           {}, {&cmd})
{
}

bool VolumeRenderPlugin::is_active(const vkb::CommandParser &parser)
{
	return true;        // FORCE ACTIVE
	                    //return parser.contains(&cmd);
}

void VolumeRenderPlugin::init(const vkb::CommandParser &parser)
{
	imin     = parser.contains(&imin_flag) ? parser.as<float>(&imin_flag) : 0.1f;
	imax     = parser.contains(&imax_flag) ? parser.as<float>(&imax_flag) : 1.0f;
	gmin     = parser.contains(&gmin_flag) ? parser.as<float>(&gmin_flag) : 0.0f;
	gmax     = parser.contains(&gmax_flag) ? parser.as<float>(&gmax_flag) : 0.2f;
	skipmode = VolumeRenderSubpass::SkippingType::Distance;
	if (parser.contains(&skipmode_flag))
	{
		uint32_t skipmode_read = parser.as<uint32_t>(&skipmode_flag);
		if (skipmode_read <= 3)
		{
			skipmode = static_cast<VolumeRenderSubpass::SkippingType>(skipmode_read);
		}
	}
	blocksize     = parser.contains(&blocksize_flag) ? parser.as<uint32_t>(&blocksize_flag) : 4;
	gradient_test = parser.contains(&gradient_test_flag);
	datasets      = {parser.contains(&dataset_flag) ? parser.as<std::string>(&dataset_flag) : "stag_beetle_832x832x494.uint16"};
	// FIXME: vkb::FlagType::ManyValues didn't seem to be working, switch to single dataset only for now
}

VolumeRender::VolumeRender() :
    camera(nullptr),
    render_sponza_scene(false),
    spin_volumes(false)
{
	//set_usage(
	//    R"(Volume renderer.
	//Usage:
	//   vulkan_samples -h | --help
	//	vulkan_samples [--imin=<arg>] [--imax=<arg>] [--gmin=<arg>] [--gmax=<arg>] [--benchmark=<frames>] [--width=<arg>] [--height=<arg>] [--skipmode=<arg>] [--blocksize=<arg>] [--gradient_test] <binary_volume_image>...)"
	//    "\n");
}

bool VolumeRender::prepare(vkb::Platform &platform)
{
	if (!Application::prepare(platform))
	{
		return false;
	}

	auto &plugin = *platform.get_plugin<VolumeRenderPlugin>();

	//std::string name        = "Volume renderer.";
	//std::string description = R"(Volume renderer.
	//Usage:
	//   vulkan_samples -h | --help
	//	vulkan_samples [--imin=<arg>] [--imax=<arg>] [--gmin=<arg>] [--gmax=<arg>] [--benchmark=<frames>] [--width=<arg>] [--height=<arg>] [--skipmode=<arg>] [--blocksize=<arg>] [--gradient_test] <binary_volume_image>...)"
	//                          "\n";
	//vkb::CLI11CommandParser parser(name, description, platform.get_arguments());

	// Override logging
	std::vector<spdlog::sink_ptr> sinks;
	sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	sinks.push_back(std::make_shared<spdlog::sinks::msvc_sink_mt>());
#endif
	//sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("log.txt", true));
	auto logger = std::make_shared<spdlog::logger>("logger", sinks.begin(), sinks.end());
	logger->set_pattern(LOGGER_FORMAT);
	spdlog::set_default_logger(logger);

	get_debug_info().insert<field::MinMax, float>("fps", fps);
	get_debug_info().insert<field::MinMax, float>("frame_time", frame_time);

	LOGI("Initializing context");

	bool     headless    = platform.get_window().get_window_mode() == Window::Mode::Headless;
	uint32_t api_version = VK_API_VERSION_1_1;
	vkb::GLSLCompiler::set_target_environment(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);

	// Creating the vulkan instance
	add_instance_extension(platform.get_surface_extension());
	instance = std::make_unique<Instance>(get_name(), get_instance_extensions(), get_validation_layers(), headless, api_version);

	// Getting a valid vulkan surface from the platform
	surface = platform.get_window().create_surface(*instance);

	auto &gpu = instance->get_suitable_gpu(surface);
	//gpu.set_high_priority_graphics_queue_enable(true);

	// Get supported features from the physical device, and requested features from the sample
	request_gpu_features(gpu);

	// Creating vulkan device, specifying the swapchain extension always
	auto requested_device_extensions = get_device_extensions();
	if (!headless || instance->is_enabled(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME))
	{
		add_device_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	}

	device = std::make_unique<vkb::Device>(gpu, surface, get_device_extensions());

	// Preparing render context for rendering
	create_render_context(platform);
	prepare_render_context();

	// Prepare compute
	compute_distance_map         = std::make_unique<ComputeDistanceMap>(*render_context);
	compute_gradient_map         = std::make_unique<ComputeGradientMap>(*render_context);
	compute_occupied_voxel_count = std::make_unique<ComputeOccupiedVoxelCount>(*render_context);

	// Load scene and camera
	load_scene("scenes/sponza/Sponza01.gltf");        // default scene
	auto &camera_node = add_free_camera(*scene, "main_camera", get_render_context().get_surface_extent());
	//auto &camera_node = add_orbit_camera("main_camera");
	camera = &camera_node.get_component<vkb::sg::Camera>();

	// Get input volumetric image filenames

	// Set volume rendering options
	volume_render_options.skipping_type = plugin.skipmode;
	if (platform.using_plugin<::plugins::BenchmarkMode>())
	{
		volume_render_options.clip_distance         = 1.0f;
		volume_render_options.early_ray_termination = false;
		volume_render_options.test                  = VolumeRenderSubpass::Test::NumTextureSamples;
		// TEST: Set camera to orthographic
	}

	// Load all of the volumes
	for (auto volume_fn : plugin.datasets)
	{
		auto volume = std::make_unique<Volume>(volume_fn);

		// Set transfer function and update distance map
		volume->options.intensity_min            = plugin.imin;
		volume->options.intensity_max            = plugin.imax;
		volume->options.gradient_min             = plugin.gmin;
		volume->options.gradient_max             = plugin.gmax;
		volume->options.use_precomputed_gradient = !plugin.gradient_test;

		// Load from disk and prep textures
		volume->load_from_file(*render_context, vkb::fs::path::get(vkb::fs::path::Assets, volume_fn), plugin.blocksize);

		auto &device = render_context->get_device();

		// Compute gradient
		if (volume->options.use_precomputed_gradient)
		{
			auto                  transfer_function_uniform = volume->get_transfer_function_uniform();
			vkb::core::Buffer     b_tf_uniform(device, sizeof(transfer_function_uniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
			vkb::BufferAllocation a_tf_uniform(b_tf_uniform, b_tf_uniform.get_size(), 0);
			b_tf_uniform.update(&transfer_function_uniform, sizeof(transfer_function_uniform));

			const auto start          = std::chrono::system_clock::now();
			auto &     command_buffer = compute_start();
			compute_gradient_map->compute(command_buffer, *volume, a_tf_uniform);
			compute_submit(command_buffer);
			const std::chrono::duration<float, std::milli> dur = std::chrono::system_clock::now() - start;
			LOGI("Updated gradient map in {}ms", dur.count());
		}

		update_transfer_function(*volume);

		// Add volume component to scene
		auto node = std::make_unique<vkb::sg::Node>(123, volume_fn);
		node->set_component(*volume);
		float scale_factor = 1.0f;
		if (platform.using_plugin<::plugins::BenchmarkMode>())
		{
			// Set scale to take up entire viewport
			glm::vec3 translation, scale, skew;
			glm::vec4 perspective;
			glm::quat rotation;
			glm::decompose(volume->get_image_transform(), scale, rotation, translation, skew, perspective);
			scale = glm::abs(rotation * glm::vec4(scale, 0.0f));
			// scale *= sqrt(3.0f);        // fits in view with arbitrary rotation
			node->get_transform().set_scale(glm::vec3(100.0f * scale_factor) / scale);
		}
		else
		{
			node->get_transform().set_scale(glm::vec3(100.0f * scale_factor));
		}
		volume->set_node(*node);
		scene->add_node(std::move(node));
		scene->add_component(std::move(volume));
	}

	// Init render pipeline
	init_render_pipeline();

	// Add a GUI with the stats you want to monitor
	//stats = std::make_unique<vkb::Stats>(std::set<vkb::StatIndex>{vkb::StatIndex::frame_times});
	stats = std::make_unique<vkb::Stats>(*render_context);
	stats->request_stats({vkb::StatIndex::frame_times});
	gui = std::make_unique<vkb::Gui>(*this, platform.get_window(), stats.get());

	return true;
}

void VolumeRender::update(float delta_time)
{
	if (spin_volumes)
	{
		auto volumes = scene->get_components<Volume>();
		for (auto volume : volumes)
		{
			// Spin volumes
			auto &transform = volume->get_node()->get_transform();
			auto  rotation  = transform.get_rotation();
			transform.set_rotation(glm::rotate(rotation, glm::radians(90.0f) * delta_time, glm::vec3(0, 1, 0)));
		}
	}

	VulkanSample::update(delta_time);
}

/**
* @return Load store info to clear all and store only the swapchain
*/
std::vector<vkb::LoadStoreInfo> get_clear_all_store_swapchain()
{
	// Clear every attachment and store only swapchain
	std::vector<vkb::LoadStoreInfo> load_store{2};

	// Swapchain
	load_store[0].load_op  = VK_ATTACHMENT_LOAD_OP_CLEAR;
	load_store[0].store_op = VK_ATTACHMENT_STORE_OP_STORE;

	// Depth
	load_store[1].load_op  = VK_ATTACHMENT_LOAD_OP_CLEAR;
	load_store[1].store_op = VK_ATTACHMENT_STORE_OP_STORE;

	return load_store;
}

vkb::CommandBuffer &VolumeRender::compute_start()
{
	render_context->begin_frame();
	auto &queue          = render_context->get_device().get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0);
	auto &command_buffer = render_context->get_active_frame().request_command_buffer(queue);
	command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	return command_buffer;
}

void VolumeRender::compute_submit(vkb::CommandBuffer &command_buffer)
{
	command_buffer.end();

	auto &device = render_context->get_device();

	auto &queue = device.get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0);

	const VkSemaphore signal_semaphores[] = {
	    render_context->request_semaphore(),
	};

	auto info                 = vkb::initializers::submit_info();
	info.pSignalSemaphores    = signal_semaphores;
	info.signalSemaphoreCount = 1;
	//info.pWaitSemaphores      = wait_semaphores;
	info.waitSemaphoreCount = 0;
	//info.pWaitDstStageMask    = wait_stages;
	info.commandBufferCount = 1;
	info.pCommandBuffers    = &command_buffer.get_handle();

	auto fence = render_context->get_active_frame().request_fence();
	queue.submit({info}, fence);
	VK_CHECK(vkWaitForFences(device.get_handle(), 1, &fence, VK_TRUE, UINT64_MAX));

	render_context->end_frame(signal_semaphores[0]);
}

void VolumeRender::init_render_pipeline()
{
	// Render pipeline
	auto render_pipeline = vkb::RenderPipeline();

	if (render_sponza_scene)
	{
		// Standard scene subpass
		vkb::ShaderSource vert_shader("base.vert");
		vkb::ShaderSource frag_shader("base.frag");
		auto              scene_subpass = std::make_unique<vkb::ForwardSubpass>(*render_context, std::move(vert_shader), std::move(frag_shader), *scene, *camera);
		render_pipeline.add_subpass(std::move(scene_subpass));
	}

	// Add volume renderer subpass
	volume_render_options.depth_attachment = render_sponza_scene;
	auto volume_subpass                    = std::make_unique<VolumeRenderSubpass>(*render_context, *scene, *camera, volume_render_options);
	if (volume_render_options.depth_attachment)
	{
		volume_subpass->set_input_attachments({1});
		// FIXME: Ideally depth would be an input and an output here, but Vulkan-Samples would need a few modifications for depth to be VK_IMAGE_LAYOUT_GENERAL and for correct memory barriers
	}
	render_pipeline.add_subpass(std::move(volume_subpass));

	render_pipeline.set_load_store(get_clear_all_store_swapchain());

	set_render_pipeline(std::move(render_pipeline));
}

void VolumeRender::request_gpu_features(vkb::PhysicalDevice &gpu)
{
	gpu.get_mutable_requested_features().shaderClipDistance = gpu.get_features().shaderClipDistance;
	gpu.get_mutable_requested_features().shaderInt64        = gpu.get_features().shaderInt64;
	gpu.get_mutable_requested_features().shaderFloat64      = gpu.get_features().shaderFloat64;
}

void VolumeRender::prepare_render_context()
{
	get_render_context().prepare(1, [this](vkb::core::Image &&swapchain_image) { return create_render_target(std::move(swapchain_image)); });
}

std::unique_ptr<vkb::RenderTarget> VolumeRender::create_render_target(vkb::core::Image &&swapchain_image)
{
	auto &device = swapchain_image.get_device();
	auto &extent = swapchain_image.get_extent();

	vkb::core::Image depth_image{device,
	                             extent,
	                             VK_FORMAT_D32_SFLOAT,
	                             VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
	                             VMA_MEMORY_USAGE_GPU_ONLY};

	std::vector<vkb::core::Image> images;

	// Attachment 0
	images.push_back(std::move(swapchain_image));

	// Attachment 1
	images.push_back(std::move(depth_image));

	return std::make_unique<vkb::RenderTarget>(std::move(images));
}

void VolumeRender::update_transfer_function(Volume &volume)
{
	auto                  transfer_function_uniform = volume.get_transfer_function_uniform();
	vkb::core::Buffer     b_tf_uniform(render_context->get_device(), sizeof(transfer_function_uniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
	vkb::BufferAllocation a_tf_uniform(b_tf_uniform, b_tf_uniform.get_size(), 0);
	b_tf_uniform.update(&transfer_function_uniform, sizeof(transfer_function_uniform));

	if (platform->using_plugin<::plugins::BenchmarkMode>())
	{
		// Get buffer for computing number of occupied voxels
		vkb::core::Buffer     buffer_occupied_voxel_count = compute_occupied_voxel_count->initialise_buffer(render_context->get_device(), volume);
		vkb::BufferAllocation a_buffer_occupied_voxel_count(buffer_occupied_voxel_count, buffer_occupied_voxel_count.get_size(), 0);

		// Update transfer function and compute number of occupied voxels
		const auto start = std::chrono::system_clock::now();
		{
			auto &command_buffer = compute_start();
			volume.update_transfer_function_texture(command_buffer);
			compute_occupied_voxel_count->compute(command_buffer, volume, a_buffer_occupied_voxel_count, a_tf_uniform);
			compute_submit(command_buffer);
		}
		uint64_t                                       n_occupied_voxels       = compute_occupied_voxel_count->get_result(a_buffer_occupied_voxel_count);
		auto                                           extent                  = volume.get_volume().image->get_extent();
		size_t                                         n_voxels                = static_cast<size_t>(extent.width) * static_cast<size_t>(extent.height) * static_cast<size_t>(extent.depth);
		float                                          percent_occupied_voxels = 100.0f * static_cast<float>(n_occupied_voxels) / static_cast<float>(n_voxels);
		const std::chrono::duration<float, std::milli> dur                     = std::chrono::system_clock::now() - start;
		LOGI("Occupied voxels: {}% in {}ms", percent_occupied_voxels, dur.count());

		// Update occupancy map
		const auto start2 = std::chrono::system_clock::now();
		int        runs   = 5;
		for (int i = 0; i < runs; ++i)
		{
			auto &command_buffer = compute_start();
			compute_distance_map->compute(command_buffer, volume, a_tf_uniform, volume_render_options.skipping_type);
			compute_submit(command_buffer);
		}
		const std::chrono::duration<float, std::milli> dur2 = std::chrono::system_clock::now() - start2;
		LOGI("Updated occupancy/distance map in {}ms", dur2.count() / static_cast<float>(runs));
	}
	else
	{
		{
			auto &command_buffer = compute_start();
			volume.update_transfer_function_texture(command_buffer);
			compute_submit(command_buffer);
		}
		{
			auto &command_buffer = compute_start();
			compute_distance_map->compute(command_buffer, volume, a_tf_uniform, volume_render_options.skipping_type);
			compute_submit(command_buffer);
		}
	}
}

void VolumeRender::draw_gui()
{
	auto volumes = scene->get_components<Volume>();
	gui->show_options_window(
	    /* body = */ [this, &volumes]() {
		    auto gap = []() {
			    ImGui::SameLine();
			    ImGui::Dummy(ImVec2(20.0f, 0.0f));
			    ImGui::SameLine();
		    };

		    for (auto volume : volumes)
		    {
			    ImGui::PushID(volume->get_name().c_str());
			    ImGui::Text("%s", volume->get_name().c_str());
			    gap();
			    auto &transform   = volume->get_node()->get_transform();
			    auto  translation = transform.get_translation();
			    ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.15f);
			    if (ImGui::DragFloat3("XYZ", &translation.x, 0.1f))
			    {
				    transform.set_translation(translation);
			    }
			    ImGui::PopItemWidth();
			    gap();
			    ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.1f);
			    ImGui::SliderFloat("Sampling", &volume->options.sampling_factor, 0.5f, 3.0f, "%.3f", 2.0f);
			    gap();
			    ImGui::SliderFloat("Alpha", &volume->options.voxel_alpha_factor, 0.0f, 2.0f, "%.3f", 2.0f);
			    ImGui::PopItemWidth();

			    // Transfer function
			    ImGui::Text(" Transfer func:");
			    gap();
			    bool tf_changed = false;
			    ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.14f);
			    tf_changed |= ImGui::SliderFloat("##Intensity min", &volume->options.intensity_min, 0.0f, volume->options.intensity_max);
			    ImGui::SameLine();
			    tf_changed |= ImGui::SliderFloat("Intensity", &volume->options.intensity_max, volume->options.intensity_min, 1.0f);
			    gap();
			    tf_changed |= ImGui::SliderFloat("##Gradient min", &volume->options.gradient_min, 0.0f, volume->options.gradient_max);
			    ImGui::SameLine();
			    tf_changed |= ImGui::SliderFloat("Gradient", &volume->options.gradient_max, volume->options.gradient_min, 1.0f);
			    ImGui::PopItemWidth();

			    if (tf_changed)
			    {
				    update_transfer_function(*volume);
			    }
			    ImGui::PopID();
		    }

		    // Shared volume rendering options
		    bool changed = false;
		    ImGui::Text("ESS method:");
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("Distance (Anisotropic)", reinterpret_cast<int *>(&volume_render_options.skipping_type), static_cast<int>(VolumeRenderSubpass::SkippingType::AnisotropicDistance));
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("Distance", reinterpret_cast<int *>(&volume_render_options.skipping_type), static_cast<int>(VolumeRenderSubpass::SkippingType::Distance));
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("Block", reinterpret_cast<int *>(&volume_render_options.skipping_type), static_cast<int>(VolumeRenderSubpass::SkippingType::Block));
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("None##skipping", reinterpret_cast<int *>(&volume_render_options.skipping_type), static_cast<int>(VolumeRenderSubpass::SkippingType::None));
		    gap();

		    if (changed)
		    {
			    for (auto volume : volumes)
			    {
				    update_transfer_function(*volume);
			    }
		    }

		    changed |= ImGui::Checkbox("ERT", &volume_render_options.early_ray_termination);
		    gap();
		    ImGui::PushItemWidth(ImGui::GetWindowSize().x * 0.1f);
		    changed |= ImGui::SliderFloat("Clip dist", &volume_render_options.clip_distance, 5.0f, 500.0f, "%.3f", 2.0f);
		    ImGui::PopItemWidth();

		    // Tests
		    changed |= ImGui::Checkbox("Render sponza scene", &render_sponza_scene);
		    gap();
		    ImGui::Checkbox("Spin volumes", &spin_volumes);
		    gap();
		    ImGui::Text("Test:");
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("None##test", reinterpret_cast<int *>(&volume_render_options.test), static_cast<int>(VolumeRenderSubpass::Test::None));
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("Entry", reinterpret_cast<int *>(&volume_render_options.test), static_cast<int>(VolumeRenderSubpass::Test::RayEntry));
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("Exit", reinterpret_cast<int *>(&volume_render_options.test), static_cast<int>(VolumeRenderSubpass::Test::RayExit));
		    ImGui::SameLine();
		    changed |= ImGui::RadioButton("NumSamples", reinterpret_cast<int *>(&volume_render_options.test), static_cast<int>(VolumeRenderSubpass::Test::NumTextureSamples));

		    if (changed)
		    {
			    init_render_pipeline();
		    }
	    },
	    /* lines = */ static_cast<uint32_t>(2 + 2 * volumes.size()));
}

std::unique_ptr<vkb::VulkanSample> create_volume_render()
{
	return std::make_unique<VolumeRender>();
}
