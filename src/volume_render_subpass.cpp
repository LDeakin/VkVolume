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

#include "volume_render_subpass.h"

#include <glm/gtc/matrix_inverse.hpp>

#include "platform/filesystem.h"
#include "scene_graph/node.h"

using namespace vkb;

template <typename T>
core::Buffer stage_buffer(vkb::CommandBuffer &command_buffer, const std::vector<T> &data, vkb::core::Buffer &buffer_dst)
{
	core::Buffer stage_buffer{command_buffer.get_device(),
	                          data.size() * sizeof(T),
	                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                          VMA_MEMORY_USAGE_CPU_ONLY};

	stage_buffer.update(reinterpret_cast<const uint8_t *>(data.data()), stage_buffer.get_size());
	command_buffer.copy_buffer(stage_buffer, buffer_dst, stage_buffer.get_size());

	return stage_buffer;
}

template <typename T>
T rndUp(T a, T b)
{
	return (a + b - 1) / b;
}

VolumeRenderSubpass::VolumeRenderSubpass(RenderContext &render_context, sg::Scene &scene, sg::Camera &cam, Options options) :
    Subpass{render_context,
            {"volume_render_clipped.vert"},
            {"volume_render.frag"}},
    vertex_source_plane_intersection("volume_render_plane_intersection.vert"),
    camera{cam},
    volumes{scene.get_components<Volume>()},
    options(options)
{
	shader_variant.add_define("PRECOMPUTED_GRADIENT");
	if (options.skipping_type == SkippingType::AnisotropicDistance)
	{
		shader_variant.add_define("ANISOTROPIC_DISTANCE");
	}
	else if (options.skipping_type == SkippingType::Block)
	{
		shader_variant.add_define("BLOCK_SKIP");
	}
	else if (options.skipping_type == SkippingType::None)
	{
		shader_variant.add_define("DISABLE_SKIP");
	}
	if (!options.early_ray_termination)
	{
		shader_variant.add_define("DISABLE_EARLY_RAY_TERMINATION");
	}
	if (options.depth_attachment)
	{
		shader_variant.add_define("DEPTH_ATTACHMENT");
	}
	if (options.test == Test::RayEntry)
	{
		shader_variant.add_define("SHOW_RAY_ENTRY");
	}
	if (options.test == Test::RayExit)
	{
		shader_variant.add_define("SHOW_RAY_EXIT");
	}
	if (options.test == Test::NumTextureSamples)
	{
		shader_variant.add_define("SHOW_NUM_SAMPLES");
	}
}

void VolumeRenderSubpass::prepare()
{
	// Build all shaders upfront
	auto &resource_cache = render_context.get_device().get_resource_cache();
	resource_cache.request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, get_vertex_shader(), shader_variant);
	resource_cache.request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, vertex_source_plane_intersection, shader_variant);
	resource_cache.request_shader_module(VK_SHADER_STAGE_FRAGMENT_BIT, get_fragment_shader(), shader_variant);

	auto &device = render_context.get_device();

	// Create vertex and index buffers
	auto &command_buffer = device.request_command_buffer();
	command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	std::vector<core::Buffer> transient_buffers;

	{
		auto vertices = std::vector<glm::vec3>({{-0.5f, -0.5f, -0.5f},
		                                        {-0.5f, -0.5f, 0.5f},
		                                        {-0.5f, 0.5f, -0.5f},
		                                        {-0.5f, 0.5f, 0.5f},
		                                        {0.5f, -0.5f, -0.5f},
		                                        {0.5f, -0.5f, 0.5f},
		                                        {0.5f, 0.5f, -0.5f},
		                                        {0.5f, 0.5f, 0.5f}});
		vertex_buffer = std::make_unique<core::Buffer>(device,
		                                               vertices.size() * sizeof(glm::vec3),
		                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		                                               VMA_MEMORY_USAGE_GPU_ONLY);
		transient_buffers.push_back(stage_buffer(command_buffer, vertices, *vertex_buffer));
	}

	{
		std::vector<uint32_t> indices = {3, 0, 1, 7, 2, 3, 5, 6, 7, 1, 4, 5, 2, 4, 0, 7, 1, 5,
		                                 3, 2, 0, 7, 6, 2, 5, 4, 6, 1, 0, 4, 2, 6, 4, 7, 3, 1};
		index_count                   = static_cast<uint32_t>(indices.size());
		index_buffer                  = std::make_unique<core::Buffer>(device,
                                                      indices.size() * sizeof(uint32_t),
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                      VMA_MEMORY_USAGE_GPU_ONLY);
		transient_buffers.push_back(stage_buffer(command_buffer, indices, *index_buffer));
	}

	{
		std::vector<uint32_t> indices   = {0, 2, 1, 0, 5, 2, 4, 2, 5, 2, 4, 3};
		index_count_plane_intersection  = static_cast<uint32_t>(indices.size());
		index_buffer_plane_intersection = std::make_unique<core::Buffer>(device,
		                                                                 indices.size() * sizeof(uint32_t),
		                                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		                                                                 VMA_MEMORY_USAGE_GPU_ONLY);
		transient_buffers.push_back(stage_buffer(command_buffer, indices, *index_buffer_plane_intersection));
	}

	command_buffer.end();

	auto &queue = device.get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0);
	queue.submit(command_buffer, device.request_fence());

	device.get_fence_pool().wait();
	device.get_fence_pool().reset();
	device.get_command_pool().reset_pool();

	transient_buffers.clear();
}

void VolumeRenderSubpass::draw(CommandBuffer &command_buffer)
{
	camera.get_node()->get_transform().get_world_matrix();        // calls update_world_transform

	// Get shaders from cache
	auto &resource_cache                        = command_buffer.get_device().get_resource_cache();
	auto &vert_shader_module                    = resource_cache.request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, get_vertex_shader(), shader_variant);
	auto &vert_shader_module_plane_intersection = resource_cache.request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, vertex_source_plane_intersection, shader_variant);
	auto &frag_shader_module                    = resource_cache.request_shader_module(VK_SHADER_STAGE_FRAGMENT_BIT, get_fragment_shader(), shader_variant);

	std::vector<ShaderModule *> shader_modules{&vert_shader_module, &frag_shader_module};
	std::vector<ShaderModule *> shader_modules_plane_intersection{&vert_shader_module_plane_intersection, &frag_shader_module};

	// Create pipeline layout and bind it
	auto &pipeline_layout                    = resource_cache.request_pipeline_layout(shader_modules);
	auto &pipeline_layout_plane_intersection = resource_cache.request_pipeline_layout(shader_modules_plane_intersection);

	// Enable alpha blending
	ColorBlendAttachmentState color_blend_attachment{};
	color_blend_attachment.blend_enable = VK_TRUE;
	//color_blend_attachment.src_color_blend_factor = VK_BLEND_FACTOR_SRC_ALPHA;
	color_blend_attachment.dst_color_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	color_blend_attachment.src_alpha_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

	ColorBlendState color_blend_state{};
	color_blend_state.attachments.resize(get_output_attachments().size());
	color_blend_state.attachments[0] = color_blend_attachment;
	command_buffer.set_color_blend_state(color_blend_state);

	DepthStencilState depth_stencil_state{};
	depth_stencil_state.depth_compare_op = VK_COMPARE_OP_GREATER_OR_EQUAL;
	command_buffer.set_depth_stencil_state(depth_stencil_state);

	// Get image views of the attachments
	auto &render_target = get_render_context().get_active_frame().get_render_target();
	auto &target_views  = render_target.get_views();

	// Bind depth as input attachment
	auto &depth_view = target_views.at(1);
	command_buffer.bind_input(depth_view, 0, 0, 0);

	// Set cull mode to back
	RasterizationState rasterization_state;
	rasterization_state.cull_mode = VK_CULL_MODE_BACK_BIT;
	command_buffer.set_rasterization_state(rasterization_state);

	// Vertex input state
	VkVertexInputBindingDescription vertex_input_binding{};
	vertex_input_binding.stride = to_u32(sizeof(glm::vec3));

	// Location 0: Position
	VkVertexInputAttributeDescription pos_attr{};
	pos_attr.format = VK_FORMAT_R32G32B32_SFLOAT;
	pos_attr.offset = 0;

	VertexInputState vertex_input_state{};
	vertex_input_state.bindings   = {vertex_input_binding};
	vertex_input_state.attributes = {pos_attr};
	command_buffer.set_vertex_input_state(vertex_input_state);

	for (auto volume : volumes)
	{
		TransferFunctionUniform transfer_function_uniform = volume->get_transfer_function_uniform();

		CameraUniform camera_uniform;
		camera_uniform.camera_view          = camera.get_view();
		camera_uniform.camera_proj          = vkb::vulkan_style_projection(camera.get_projection());
		camera_uniform.camera_view_proj_inv = glm::inverse(camera_uniform.camera_proj * camera_uniform.camera_view);
		camera_uniform.model                = volume->get_node()->get_transform().get_matrix() * volume->get_image_transform();
		camera_uniform.model_inv            = glm::inverse(camera_uniform.model);

		RayCastUniform  ray_cast_uniform;
		glm::mat4       model_to_tex    = glm::translate(glm::vec3(0.5f));        // we just use a unit cube from [-0.5 to 0.5] so putting it at [0-1] is just a translation
		glm::mat4       global_to_tex   = model_to_tex * camera_uniform.model_inv;
		const glm::mat4 viewInv         = glm::inverse(camera.get_view());
		const glm::vec3 cam_pos_global  = viewInv[3];
		const glm::vec3 cam_pos_model   = camera_uniform.model_inv * glm::vec4(cam_pos_global, 1.0f);
		ray_cast_uniform.camera_pos_tex = model_to_tex * glm::vec4(cam_pos_model, 1.0f);
		const glm::vec3 cam_dir_global  = glm::vec3(viewInv * glm::vec4(0, 0, -1, 0));
		ray_cast_uniform.plane          = glm::vec4(cam_dir_global, -options.clip_distance - glm::dot(cam_pos_global, cam_dir_global));
		ray_cast_uniform.plane_tex      = glm::inverseTranspose(global_to_tex) * ray_cast_uniform.plane;
		ray_cast_uniform.front_index    = (ray_cast_uniform.plane_tex.x < 0 ? 1 : 0) +
		                               (ray_cast_uniform.plane_tex.y < 0 ? 2 : 0) +
		                               (ray_cast_uniform.plane_tex.z < 0 ? 4 : 0);
		auto volume_extent          = volume->get_volume().image->get_extent();
		auto map_extent             = volume->get_distance_map().image->get_extent();
		ray_cast_uniform.block_size = glm::vec4(
		    rndUp(volume_extent.width, map_extent.width),
		    rndUp(volume_extent.height, map_extent.height),
		    rndUp(volume_extent.depth, map_extent.depth),
		    0);
		//options.resume_factor * transfer_function_uniform.sampling_factor *
		//std::min(std::min(ray_cast_uniform.block_size.x, ray_cast_uniform.block_size.y), ray_cast_uniform.block_size.z);

		// Allocate a buffer using the buffer pool from the active frame to store uniform values and bind it
		auto &render_frame                 = get_render_context().get_active_frame();
		auto  allocation_transfer_function = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(transfer_function_uniform));
		allocation_transfer_function.update(transfer_function_uniform);
		auto allocation_camera = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(camera_uniform));
		allocation_camera.update(camera_uniform);
		auto allocation_ray_cast = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(ray_cast_uniform));
		allocation_ray_cast.update(ray_cast_uniform);

		// Draw clipped cuboid
		command_buffer.bind_pipeline_layout(pipeline_layout);
		command_buffer.bind_buffer(allocation_camera.get_buffer(), allocation_camera.get_offset(), allocation_camera.get_size(), 0, 1, 0);
		command_buffer.bind_buffer(allocation_ray_cast.get_buffer(), allocation_ray_cast.get_offset(), allocation_ray_cast.get_size(), 0, 2, 0);
		command_buffer.bind_buffer(allocation_transfer_function.get_buffer(), allocation_transfer_function.get_offset(), allocation_transfer_function.get_size(), 0, 3, 0);
		command_buffer.bind_image(*volume->get_transfer_function().image_view, *volume->get_transfer_function().sampler, 0, 4, 0);
		command_buffer.bind_image(*volume->get_volume().image_view, *volume->get_volume().sampler, 0, 5, 0);
		command_buffer.bind_image(*volume->get_gradient().image_view, *volume->get_gradient().sampler, 0, 6, 0);
		if (options.skipping_type == SkippingType::AnisotropicDistance)
		{
			for (int i = 0; i < 8; ++i)
			{
				auto &distance_map = volume->get_distance_map(i);
				command_buffer.bind_image(*distance_map.image_view, *distance_map.sampler, 0, 7, i);
			}
		}
		else
		{
			command_buffer.bind_image(*volume->get_distance_map().image_view, *volume->get_distance_map().sampler, 0, 7, 0);
		}
		command_buffer.bind_vertex_buffers(0, {*vertex_buffer}, {0});
		command_buffer.bind_index_buffer(*index_buffer, 0, VkIndexType::VK_INDEX_TYPE_UINT32);
		command_buffer.draw_indexed(index_count, 1, 0, 0, 0);

		// Draw box plane intersection
		command_buffer.bind_pipeline_layout(pipeline_layout_plane_intersection);
		command_buffer.bind_index_buffer(*index_buffer_plane_intersection, 0, VkIndexType::VK_INDEX_TYPE_UINT32);
		command_buffer.draw_indexed(index_count_plane_intersection, 1, 0, 0, 0);
	}
}