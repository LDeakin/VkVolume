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

#include "gltf_loader.h"
#include "gui.h"
#include "platform/filesystem.h"
#include "platform/platform.h"
#include "rendering/render_context.h"
#include "rendering/render_pipeline.h"
#include "rendering/subpasses/forward_subpass.h"
#include "scene_graph/components/perspective_camera.h"
#include "stats.h"

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

#include "orbit_camera.h"
#include "volume_render_subpass.h"

using namespace vkb;

VolumeRender::VolumeRender() :
    camera(nullptr),
    block_size(4),
    render_sponza_scene(false),
    spin_volumes(false)
{
	set_usage(
	    R"(Volume renderer.
	Usage:
    vulkan_samples -h | --help
		vulkan_samples [--imin=<arg>] [--imax=<arg>] [--gmin=<arg>] [--gmax=<arg>] [--benchmark=<frames>] [--width=<arg>] [--height=<arg>] [--skipmode=<arg>] [--blocksize=<arg>] [--gradient_test] <binary_volume_image>...)"
	    "\n");
}

sg::Node &VolumeRender::add_orbit_camera(const std::string &node_name)
{
	auto camera_node = scene->find_node(node_name);

	if (!camera_node)
	{
		LOGW("Camera node `{}` not found. Looking for `default_camera` node.", node_name.c_str());

		camera_node = scene->find_node("default_camera");
	}

	if (!camera_node)
	{
		throw std::runtime_error("Camera node with name `" + node_name + "` not found.");
	}

	if (!camera_node->has_component<sg::Camera>())
	{
		throw std::runtime_error("No camera component found for `" + node_name + "` node.");
	}

	auto orbit_camera_script = std::make_unique<OrbitCamera>(*camera_node);
	if (is_benchmark_mode())
	{
		orbit_camera_script->zoom_ = -100.0f * sqrt(3.0f);
		orbit_camera_script->recalculate_view();
	}
	orbit_camera_script->resize(render_context->get_surface_extent().width, render_context->get_surface_extent().height);

	scene->add_component(std::move(orbit_camera_script), *camera_node);

	return *camera_node;
}

bool VolumeRender::prepare(vkb::Platform &platform)
{
	if (!Application::prepare(platform))
	{
		return false;
	}

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

	// Creating the vulkan instance
	std::vector<const char *> requested_instance_extensions = get_instance_extensions();
	requested_instance_extensions.push_back(platform.get_surface_extension());
	instance = std::make_unique<Instance>(get_name(), requested_instance_extensions, get_validation_layers(), is_headless());

	// Getting a valid vulkan surface from the platform
	surface = platform.get_window().create_surface(*instance);

	auto physical_device = instance->get_gpu();

	// Get supported features from the physical device, and requested features from the sample
	vkGetPhysicalDeviceFeatures(physical_device, &supported_device_features);
	get_device_features();

	// Creating vulkan device, specifying the swapchain extension always
	std::vector<const char *> requested_device_extensions = get_device_extensions();
	if (!is_headless() || instance->is_enabled(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME))
	{
		requested_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	}

	device = std::make_unique<vkb::Device>(physical_device, surface, requested_device_extensions, requested_device_features);

	// Preparing render context for rendering
	render_context = std::make_unique<vkb::RenderContext>(*device, surface, platform.get_window().get_width(), platform.get_window().get_height());
	render_context->set_present_mode_priority({VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR,
	                                           VK_PRESENT_MODE_FIFO_KHR});

	render_context->set_surface_format_priority({{VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
	                                             {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
	                                             {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
	                                             {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}});

	render_context->request_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);        // disables vsync

	prepare_render_context();

	// Prepare compute
	compute_distance_map         = std::make_unique<ComputeDistanceMap>(*render_context);
	compute_gradient_map         = std::make_unique<ComputeGradientMap>(*render_context);
	compute_occupied_voxel_count = std::make_unique<ComputeOccupiedVoxelCount>(*render_context);

	// Load scene and camera
	load_scene("scenes/sponza/Sponza01.gltf");        // default scene
	//auto &camera_node = add_free_camera("main_camera");
	auto &camera_node = add_orbit_camera("main_camera");
	camera            = &camera_node.get_component<vkb::sg::Camera>();

	// Get input volumetric image filenames
	std::vector<std::string> volume_fns = {"stag_beetle_832x832x494.uint16"};
	if (options.contains("<binary_volume_image>"))
	{
		volume_fns = options.get_list("<binary_volume_image>");
	}

	// Set volume rendering options
	block_size = get_options().contains("--blocksize") ? get_options().get_int("--blocksize") : block_size;
	if (get_options().contains("--skipmode"))
	{
		volume_render_options.skipping_type = (VolumeRenderSubpass::SkippingType) get_options().get_int("--skipmode");
	}
	else
	{
		volume_render_options.skipping_type = VolumeRenderSubpass::SkippingType::Distance;
	}
	if (is_benchmark_mode())
	{
		volume_render_options.clip_distance         = 1.0f;
		volume_render_options.early_ray_termination = false;
		volume_render_options.test                  = VolumeRenderSubpass::Test::NumTextureSamples;
		// TEST: Set camera to orthographic
	}

	// Load all of the volumes
	for (auto volume_fn : volume_fns)
	{
		auto volume = std::make_unique<Volume>(volume_fn);

		// Set transfer function and update distance map
		auto get_with_default = [this](std::string name, float default_value) {
			return get_options().contains(name) ? std::stof(get_options().get_string(name)) : default_value;
		};
		volume->options.intensity_min            = get_with_default("--imin", 0.1f);
		volume->options.intensity_max            = get_with_default("--imax", 1.0f);
		volume->options.gradient_min             = get_with_default("--gmin", 0.0f);
		volume->options.gradient_max             = get_with_default("--gmax", 0.2f);
		volume->options.use_precomputed_gradient = !get_options().contains("--gradient_test");

		// Load from disk and prep textures
		volume->load_from_file(*render_context, vkb::fs::path::get(vkb::fs::path::Assets, volume_fn), block_size);

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
		if (is_benchmark_mode())
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
	stats = std::make_unique<vkb::Stats>(std::set<vkb::StatIndex>{vkb::StatIndex::frame_times});
	gui   = std::make_unique<vkb::Gui>(*this, platform.get_window().get_dpi_factor());

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
	auto &command_buffer = render_context->begin_compute();
	command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	return command_buffer;
}

void VolumeRender::compute_submit(vkb::CommandBuffer &command_buffer)
{
	command_buffer.end();

	// Wait for the command buffer to finish its work before destroying the staging buffer
	auto &      device = render_context->get_device();
	const auto &queue  = render_context->get_device().get_queue_by_flags(VK_QUEUE_COMPUTE_BIT, 0);
	queue.submit(command_buffer, device.request_fence());
	device.get_fence_pool().wait();
	device.get_fence_pool().reset();
	device.get_command_pool().reset_pool();
	render_context->end_compute();
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

void VolumeRender::get_device_features()
{
	requested_device_features.shaderClipDistance = supported_device_features.shaderClipDistance;
	requested_device_features.shaderInt64        = supported_device_features.shaderInt64;
	requested_device_features.shaderFloat64      = supported_device_features.shaderFloat64;
}

void VolumeRender::prepare_render_context()
{
	get_render_context().prepare(1, std::bind(&VolumeRender::create_render_target, this, std::placeholders::_1));
}

vkb::RenderTarget VolumeRender::create_render_target(vkb::core::Image &&swapchain_image)
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

	return vkb::RenderTarget{std::move(images)};
}

void VolumeRender::update_transfer_function(Volume &volume)
{
	auto                  transfer_function_uniform = volume.get_transfer_function_uniform();
	vkb::core::Buffer     b_tf_uniform(render_context->get_device(), sizeof(transfer_function_uniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
	vkb::BufferAllocation a_tf_uniform(b_tf_uniform, b_tf_uniform.get_size(), 0);
	b_tf_uniform.update(&transfer_function_uniform, sizeof(transfer_function_uniform));

	if (is_benchmark_mode())
	{
		// Get buffer for computing number of occupied voxels
		vkb::core::Buffer     buffer_occupied_voxel_count = compute_occupied_voxel_count->initialise_buffer(render_context->get_device(), volume);
		vkb::BufferAllocation a_buffer_occupied_voxel_count(buffer_occupied_voxel_count, buffer_occupied_voxel_count.get_size(), 0);

		// Update transfer function and compute number of occupied voxels
		const auto start          = std::chrono::system_clock::now();
		auto &     command_buffer = compute_start();
		volume.update_transfer_function_texture(command_buffer);
		compute_occupied_voxel_count->compute(command_buffer, volume, a_buffer_occupied_voxel_count, a_tf_uniform);
		compute_submit(command_buffer);
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
