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

#include "compute_gradient_map.h"

#include "common/vk_common.h"
#include "platform/filesystem.h"
#include "rendering/render_context.h"

auto rndUp = [](int x, int y) { return (x + y - 1) / y; };

ComputeGradientMap::ComputeGradientMap(vkb::RenderContext &render_context) :
    render_context(render_context),
    compute_shader("gradient_map.comp")
{
	// Build all shaders upfront
	auto &resource_cache = render_context.get_device().get_resource_cache();
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader);

	// Memory barriers
	memory_barrier_to_compute.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	memory_barrier_to_compute.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_to_compute.src_access_mask = 0;
	memory_barrier_to_compute.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_to_compute.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	memory_barrier_to_compute.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

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

void ComputeGradientMap::compute(vkb::CommandBuffer &command_buffer, Volume &volume, vkb::BufferAllocation &transfer_function_uniform)
{
	// Set layout
	auto &volume_tex   = volume.get_volume();
	auto &gradient_tex = volume.get_gradient();
	command_buffer.image_memory_barrier(*volume_tex.image_view, memory_barrier_to_compute);
	command_buffer.image_memory_barrier(*gradient_tex.image_view, memory_barrier_to_compute);

	auto &resource_cache  = command_buffer.get_device().get_resource_cache();
	auto &shader_module   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader);
	auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});

	// Bind pipeline layout and images
	command_buffer.bind_pipeline_layout(pipeline_layout);
	command_buffer.bind_input(*volume_tex.image_view, 0, 0, 0);
	command_buffer.bind_buffer(transfer_function_uniform.get_buffer(), transfer_function_uniform.get_offset(), transfer_function_uniform.get_size(), 0, 1, 0);
	//command_buffer.bind_input(*transfer_function.image_view, 0, 2, 0);        // Transfer function texture, need variant TRANSFER_FUNCTION_TEXTURE
	command_buffer.bind_input(*gradient_tex.image_view, 0, 3, 0);
	auto extent = volume_tex.image->get_extent();
	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.height, 8), rndUp(extent.depth, 8));

	// Reset layout
	command_buffer.image_memory_barrier(*volume_tex.image_view, memory_barrier_compute_to_fragment);
	command_buffer.image_memory_barrier(*gradient_tex.image_view, memory_barrier_compute_to_fragment);
}
