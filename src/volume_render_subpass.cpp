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

VolumeRenderSubpass::VolumeRenderSubpass(RenderContext &render_context, sg::Scene &scene, sg::Camera &cam, Options options) :
    Subpass{render_context,
            vkb::fs::read_shader("volume_render_clipped.vert"),
            vkb::fs::read_shader("volume_render.frag")},
    vertex_source_plane_intersection(vkb::fs::read_shader("volume_render_plane_intersection.vert")),
    camera{cam},
    volumes{scene.get_components<Volume>()},
    options(options)
{
	if (options.skipping_type == SkippingType::Block)
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

	// Build all shaders upfront
	auto &resource_cache = render_context.get_device().get_resource_cache();
	resource_cache.request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, get_vertex_shader(), shader_variant);
	resource_cache.request_shader_module(VK_SHADER_STAGE_VERTEX_BIT, vertex_source_plane_intersection, shader_variant);
	resource_cache.request_shader_module(VK_SHADER_STAGE_FRAGMENT_BIT, get_fragment_shader(), shader_variant);

	// FIXME: split the uniform into global parameters, per volume parameters and per-frame parameters (which should be dynamic)
	//vert_module.set_resource_dynamic("VolumeRenderUniform");

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

	command_buffer.set_depth_stencil_state(get_depth_stencil_state());

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
		// Populate uniform values
		VolumeRenderUniform volume_render_uniform;
		volume_render_uniform.sampling_factor         = volume->options.sampling_factor;
		volume_render_uniform.voxel_alpha_factor      = volume->options.voxel_alpha_factor;
		volume_render_uniform.grad_magnitude_modifier = 1.0f;
		volume_render_uniform.intensity_min           = volume->options.intensity_min;
		volume_render_uniform.intensity_max           = volume->options.intensity_max;
		volume_render_uniform.gradient_min            = volume->options.gradient_min;
		volume_render_uniform.gradient_max            = volume->options.gradient_max;
		volume_render_uniform.camera_view             = camera.get_view();
		volume_render_uniform.camera_proj             = vkb::vulkan_style_projection(camera.get_projection());
		volume_render_uniform.camera_view_proj_inv    = glm::inverse(volume_render_uniform.camera_proj * volume_render_uniform.camera_view);
		volume_render_uniform.model                   = volume->get_node()->get_transform().get_matrix() * volume->get_image_transform();
		volume_render_uniform.model_inv               = glm::inverse(volume_render_uniform.model);

		// Coordinate transformations between global, model (local) and texture coordinates
		glm::mat4 model_to_tex  = glm::translate(glm::vec3(0.5f));        // we just use a unit cube from [-0.5 to 0.5] so putting it at [0-1] is just a translation
		glm::mat4 global_to_tex = model_to_tex * volume_render_uniform.model_inv;

		// Camera position in texture coordinates
		const glm::mat4 viewInv              = glm::inverse(camera.get_view());
		const glm::vec3 cam_pos_global       = viewInv[3];
		const glm::vec3 cam_pos_model        = volume_render_uniform.model_inv * glm::vec4(cam_pos_global, 1.0f);
		volume_render_uniform.camera_pos_tex = model_to_tex * glm::vec4(cam_pos_model, 1.0f);

		// Clip plane in global coordinates and texture coordinates
		const glm::vec3 cam_dir_global  = glm::vec3(viewInv * glm::vec4(0, 0, -1, 0));
		volume_render_uniform.plane     = glm::vec4(cam_dir_global, -options.clip_distance - glm::dot(cam_pos_global, cam_dir_global));
		volume_render_uniform.plane_tex = glm::inverseTranspose(global_to_tex) * volume_render_uniform.plane;        // plane transform is inverse transpose of equivalent vector transform

		// Get the "front" index of the cube (i.e. vertex furthest behind the clipping plane)
		volume_render_uniform.front_index = (volume_render_uniform.plane_tex.x < 0 ? 1 : 0) +
		                                    (volume_render_uniform.plane_tex.y < 0 ? 2 : 0) +
		                                    (volume_render_uniform.plane_tex.z < 0 ? 4 : 0);

		// Allocate a buffer using the buffer pool from the active frame to store uniform values and bind it
		auto &render_frame = get_render_context().get_active_frame();
		auto  allocation   = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(VolumeRenderUniform));
		allocation.update(volume_render_uniform);

		// Draw clipped cuboid
		command_buffer.bind_pipeline_layout(pipeline_layout);
		command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 1, 0);
		command_buffer.bind_image(*volume->get_volume().image_view, *volume->get_volume().sampler, 0, 2, 0);
		//command_buffer.bind_image(*volume->get_gradient().image_view, *volume->get_gradient().sampler, 0, 3, 0);
		//command_buffer.bind_image(*volume->get_transfer_function().image_view, *volume->get_transfer_function().sampler, 0, 4, 0);
		command_buffer.bind_image(*volume->get_distance_map().image_view, *volume->get_distance_map().sampler, 0, 5, 0);
		command_buffer.bind_vertex_buffers(0, {*vertex_buffer}, {0});
		command_buffer.bind_index_buffer(*index_buffer, 0, VkIndexType::VK_INDEX_TYPE_UINT32);
		command_buffer.draw_indexed(index_count, 1, 0, 0, 0);

		// Draw box plane intersection
		command_buffer.bind_pipeline_layout(pipeline_layout_plane_intersection);
		command_buffer.bind_index_buffer(*index_buffer_plane_intersection, 0, VkIndexType::VK_INDEX_TYPE_UINT32);
		command_buffer.draw_indexed(index_count_plane_intersection, 1, 0, 0, 0);
	}
}