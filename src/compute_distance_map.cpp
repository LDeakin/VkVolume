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

#include "compute_distance_map.h"

#include "common/vk_common.h"
#include "platform/filesystem.h"
#include "rendering/render_context.h"

auto rndUp = [](int x, int y) { return (x + y - 1) / y; };

ComputeDistanceMap::ComputeDistanceMap(vkb::RenderContext &render_context) :
    render_context(render_context),
    compute_shader_occupancy(vkb::fs::read_shader("occupancy_map.comp")),
    compute_shader_distance(vkb::fs::read_shader("distance_map.comp"))
{
	// Build all shaders upfront
	auto &resource_cache = render_context.get_device().get_resource_cache();
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_occupancy);
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_distance);

	// Memory barriers
	memory_barrier_fragment_to_compute.old_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	memory_barrier_fragment_to_compute.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_fragment_to_compute.src_access_mask = 0;
	memory_barrier_fragment_to_compute.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_fragment_to_compute.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	memory_barrier_fragment_to_compute.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	memory_barrier_write_to_read.old_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_write_to_read.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_write_to_read.src_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
	memory_barrier_write_to_read.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_write_to_read.src_stage_mask  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	memory_barrier_write_to_read.dst_stage_mask  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

	memory_barrier_compute_to_fragment.old_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_compute_to_fragment.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	memory_barrier_compute_to_fragment.src_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
	memory_barrier_compute_to_fragment.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_compute_to_fragment.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	memory_barrier_compute_to_fragment.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
}

void ComputeDistanceMap::compute(vkb::CommandBuffer &command_buffer, const Volume::Image &volume, const Volume::Image &distance_map,
                                 const Volume::Image &distance_map_swap, float intensity_min, float intensity_max, float gradient_min, float gradient_max)
{
	// Get shaders from cache
	auto &resource_cache            = command_buffer.get_device().get_resource_cache();
	auto &shader_module_occupancy   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_occupancy);
	auto &pipeline_layout_occupancy = resource_cache.request_pipeline_layout({&shader_module_occupancy});

	// Prepare inputs for compute shader
	command_buffer.image_memory_barrier(*volume.image_view, memory_barrier_fragment_to_compute);
	command_buffer.image_memory_barrier(*distance_map.image_view, memory_barrier_fragment_to_compute);

	// Run compute shaders
	computeOccupancy(command_buffer, volume, distance_map, intensity_min, intensity_max, gradient_min, gradient_max);
	computeDistance(command_buffer, distance_map, distance_map_swap);

	// Reset layouts
	command_buffer.image_memory_barrier(*volume.image_view, memory_barrier_compute_to_fragment);
	command_buffer.image_memory_barrier(*distance_map.image_view, memory_barrier_compute_to_fragment);
}

void ComputeDistanceMap::computeOccupancy(vkb::CommandBuffer &command_buffer, const Volume::Image &volume, const Volume::Image &occupancy_map,
                                          float intensity_min, float intensity_max, float gradient_min, float gradient_max)
{
	auto &resource_cache  = command_buffer.get_device().get_resource_cache();
	auto &shader_module   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_occupancy);
	auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});

	// Compute block size
	glm::ivec3 block_size(
	    volume.image->get_extent().width / occupancy_map.image->get_extent().width,
	    volume.image->get_extent().height / occupancy_map.image->get_extent().height,
	    volume.image->get_extent().depth / occupancy_map.image->get_extent().depth);

	struct PushConstants
	{
		glm::vec4 block_size;
		float     grad_magnitude_modifier;
		float     intensity_min;
		float     intensity_max;
		float     gradient_min;
		float     gradient_max;
	};
	PushConstants pushConstants = {glm::vec4(block_size, 0.0f), 1.0f, intensity_min, intensity_max, gradient_min, gradient_max};

	// Bind pipeline layout and images
	command_buffer.bind_pipeline_layout(pipeline_layout);
	command_buffer.bind_input(*volume.image_view, 0, 0, 0);
	//command_buffer.bind_input(*gradient.image_view, 0, 1, 0);        // Precomputed gradient if, need variant PRECOMPUTED_GRADIENT
	//command_buffer.bind_input(*transfer_function.image_view, 0, 2, 0);        // Transfer function texture, need variant TRANSFER_FUNCTION_TEXTURE
	command_buffer.bind_input(*occupancy_map.image_view, 0, 3, 0);
	command_buffer.push_constants<PushConstants>(0, pushConstants);

	auto extent = occupancy_map.image->get_extent();
	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.height, 8), rndUp(extent.depth, 8));
	command_buffer.image_memory_barrier(*occupancy_map.image_view, memory_barrier_write_to_read);
}

void ComputeDistanceMap::computeDistance(vkb::CommandBuffer &command_buffer, const Volume::Image &occupancy_and_distance_map, const Volume::Image &distance_map_swap)
{
	auto &resource_cache  = command_buffer.get_device().get_resource_cache();
	auto &shader_module   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_distance);
	auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});

	// Bind pipeline layout and images
	auto extent = occupancy_and_distance_map.image->get_extent();
	command_buffer.bind_pipeline_layout(pipeline_layout);
	command_buffer.bind_input(*occupancy_and_distance_map.image_view, 0, 0, 0);
	command_buffer.bind_input(*occupancy_and_distance_map.image_view, 0, 1, 0);

	// Dispatch 1st stage
	command_buffer.push_constants<uint32_t>(0, 0);
	command_buffer.dispatch(rndUp(extent.height, 8), rndUp(extent.depth, 8), 1);
	command_buffer.image_memory_barrier(*occupancy_and_distance_map.image_view, memory_barrier_write_to_read);

	// Dispatch 2nd stage
	command_buffer.bind_input(*distance_map_swap.image_view, 0, 1, 0);
	command_buffer.push_constants<uint32_t>(0, 1);
	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.depth, 8), 1);
	command_buffer.image_memory_barrier(*distance_map_swap.image_view, memory_barrier_write_to_read);

	// Dispatch 3rd stage
	command_buffer.push_constants<uint32_t>(0, 2);
	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.height, 8), 1);
	command_buffer.image_memory_barrier(*occupancy_and_distance_map.image_view, memory_barrier_write_to_read);
}
