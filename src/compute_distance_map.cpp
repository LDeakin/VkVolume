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
    compute_shader_occupancy("occupancy_map.comp"),
    compute_shader_distance("distance_map.comp"),
    compute_shader_distance_anisotropic("distance_map_anisotropic.comp")
{
	vkb::ShaderVariant variant;
	variant.add_define("PRECOMPUTED_GRADIENT");

	// Build all shaders upfront
	auto &resource_cache = render_context.get_device().get_resource_cache();
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_occupancy);
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_occupancy, variant);
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_distance);
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_distance_anisotropic);

	// Memory barriers
	memory_barrier_to_compute.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	memory_barrier_to_compute.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_to_compute.src_access_mask = 0;
	memory_barrier_to_compute.dst_access_mask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_to_compute.src_stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	memory_barrier_to_compute.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	memory_barrier_write_to_read.old_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_write_to_read.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_write_to_read.src_access_mask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_write_to_read.dst_access_mask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_write_to_read.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	memory_barrier_write_to_read.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	memory_barrier_compute_to_fragment.old_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_compute_to_fragment.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	memory_barrier_compute_to_fragment.src_access_mask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_compute_to_fragment.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_compute_to_fragment.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	memory_barrier_compute_to_fragment.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
}

void ComputeDistanceMap::compute(vkb::CommandBuffer &command_buffer, Volume &volume, vkb::BufferAllocation &transfer_function_uniform, VolumeRenderSubpass::SkippingType skipping_type)
{
	bool anisotropic     = skipping_type == VolumeRenderSubpass::SkippingType::AnisotropicDistance;
	int  n_distance_maps = anisotropic ? 8 : 1;
	volume.set_number_of_distance_maps(render_context, n_distance_maps);

	// Occupancy
	auto &occupancy_map = volume.get_distance_map(n_distance_maps - 1);
	command_buffer.image_memory_barrier(*occupancy_map.image_view, memory_barrier_to_compute);
	command_buffer.image_memory_barrier(*volume.get_volume().image_view, memory_barrier_to_compute);
	if (volume.options.use_precomputed_gradient)
	{
		command_buffer.image_memory_barrier(*volume.get_gradient().image_view, memory_barrier_to_compute);
	}
	computeOccupancy(command_buffer, volume, occupancy_map, transfer_function_uniform);
	if (volume.options.use_precomputed_gradient)
	{
		command_buffer.image_memory_barrier(*volume.get_gradient().image_view, memory_barrier_compute_to_fragment);
	}

	// Distance map
	command_buffer.image_memory_barrier(*volume.get_volume().image_view, memory_barrier_to_compute);
	command_buffer.image_memory_barrier(*volume.get_distance_map_swap().image_view, memory_barrier_to_compute);
	if (skipping_type == VolumeRenderSubpass::SkippingType::AnisotropicDistance)
	{
		computeDistanceAnisotropic(command_buffer, volume);
	}
	else if (skipping_type == VolumeRenderSubpass::SkippingType::Distance)
	{
		computeDistance(command_buffer, volume);
	}
	else
	{
		command_buffer.image_memory_barrier(*occupancy_map.image_view, memory_barrier_compute_to_fragment);
	}
	command_buffer.image_memory_barrier(*volume.get_volume().image_view, memory_barrier_compute_to_fragment);
}

void ComputeDistanceMap::computeOccupancy(vkb::CommandBuffer &command_buffer, const Volume &volume, const Volume::Image &occupancy_map,
                                          vkb::BufferAllocation &transfer_function_uniform)
{
	auto &volume_tex = volume.get_volume();
	// Compute block size
	auto       extent        = occupancy_map.image->get_extent();
	auto       volume_extent = volume_tex.image->get_extent();
	glm::ivec3 block_size(
	    rndUp(volume_extent.width, extent.width),
	    rndUp(volume_extent.height, extent.height),
	    rndUp(volume_extent.depth, extent.depth));

	vkb::ShaderVariant variant;
	if (volume.options.use_precomputed_gradient)
	{
		variant.add_define("PRECOMPUTED_GRADIENT");
	}

	auto &resource_cache  = command_buffer.get_device().get_resource_cache();
	auto &shader_module   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_occupancy, variant);
	auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});

	// Bind pipeline layout and images
	command_buffer.bind_pipeline_layout(pipeline_layout);
	command_buffer.bind_input(*volume_tex.image_view, 0, 0, 0);
	command_buffer.bind_buffer(transfer_function_uniform.get_buffer(), transfer_function_uniform.get_offset(), transfer_function_uniform.get_size(), 0, 1, 0);
	command_buffer.bind_image(*volume.get_transfer_function().image_view, *volume.get_transfer_function().sampler, 0, 2, 0);
	if (volume.options.use_precomputed_gradient)
	{
		command_buffer.bind_input(*volume.get_gradient().image_view, 0, 3, 0);
	}
	command_buffer.bind_input(*occupancy_map.image_view, 0, 4, 0);

	command_buffer.push_constants(glm::ivec4(block_size, 0));
	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.height, 8), rndUp(extent.depth, 8));

	command_buffer.image_memory_barrier(*occupancy_map.image_view, memory_barrier_write_to_read);
}

void ComputeDistanceMap::computeDistance(vkb::CommandBuffer &command_buffer, const Volume &volume)
{
	auto &resource_cache  = command_buffer.get_device().get_resource_cache();
	auto &shader_module   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_distance);
	auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});

	auto &distance = volume.get_distance_map();        // also the occupancy map, done in-place
	auto &swap     = volume.get_distance_map_swap();

	command_buffer.image_memory_barrier(*distance.image_view, memory_barrier_to_compute);

	// Bind pipeline layout and images
	auto extent = distance.image->get_extent();
	command_buffer.bind_pipeline_layout(pipeline_layout);
	command_buffer.bind_input(*distance.image_view, 0, 0, 0);
	command_buffer.bind_input(*distance.image_view, 0, 1, 0);

	// Dispatch 1st stage
	command_buffer.push_constants<uint32_t>(0);
	command_buffer.dispatch(rndUp(extent.height, 8), rndUp(extent.depth, 8), 1);
	command_buffer.image_memory_barrier(*distance.image_view, memory_barrier_write_to_read);

	// Dispatch 2nd stage
	command_buffer.bind_input(*swap.image_view, 0, 1, 0);
	command_buffer.push_constants<uint32_t>(1);
	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.depth, 8), 1);
	command_buffer.image_memory_barrier(*swap.image_view, memory_barrier_write_to_read);

	// Dispatch 3rd stage
	command_buffer.push_constants<uint32_t>(2);
	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.height, 8), 1);

	command_buffer.image_memory_barrier(*distance.image_view, memory_barrier_compute_to_fragment);
}

void ComputeDistanceMap::computeDistanceAnisotropic(vkb::CommandBuffer &command_buffer, const Volume &volume)
{
	auto &resource_cache  = command_buffer.get_device().get_resource_cache();
	auto &shader_module   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_distance_anisotropic);
	auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});

	auto &occupancy_map = volume.get_distance_map(7);
	auto &swap          = volume.get_distance_map_swap();
	auto  extent        = occupancy_map.image->get_extent();

	command_buffer.bind_pipeline_layout(pipeline_layout);

	for (int i = 0; i < 8; ++i)
	{
		auto &distance = volume.get_distance_map(i);
		command_buffer.image_memory_barrier(*distance.image_view, memory_barrier_to_compute);
	}

	struct PushConstants
	{
		uint32_t stage;
		int32_t  direction;
	};

	auto stage1 = [&](size_t distance_map_idx, int32_t direction) {
		auto &distance = volume.get_distance_map(distance_map_idx);
		command_buffer.push_constants<PushConstants>({0, direction});
		command_buffer.bind_input(*distance.image_view, 0, 0, 0);
		command_buffer.bind_input(*occupancy_map.image_view, 0, 1, 0);
		command_buffer.dispatch(rndUp(extent.height, 8), rndUp(extent.depth, 8), 1);
		command_buffer.image_memory_barrier(*volume.get_distance_map(distance_map_idx).image_view, memory_barrier_write_to_read);
	};

	auto stage2 = [&](size_t distance_map_idx, int32_t direction) {
		auto &distance = volume.get_distance_map(distance_map_idx);
		command_buffer.push_constants<PushConstants>({1, direction});
		command_buffer.bind_input(*distance.image_view, 0, 0, 0);
		command_buffer.bind_input(*swap.image_view, 0, 1, 0);
		command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.depth, 8), 1);
		command_buffer.image_memory_barrier(*swap.image_view, memory_barrier_write_to_read);
	};

	auto stage3 = [&](size_t distance_map_idx, int32_t direction) {
		auto &distance = volume.get_distance_map(distance_map_idx);
		command_buffer.image_memory_barrier(*distance.image_view, memory_barrier_to_compute);
		command_buffer.push_constants<PushConstants>({2, direction});
		command_buffer.bind_input(*distance.image_view, 0, 0, 0);
		command_buffer.bind_input(*swap.image_view, 0, 1, 0);
		command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.height, 8), 1);
		command_buffer.image_memory_barrier(*volume.get_distance_map(distance_map_idx).image_view, memory_barrier_write_to_read);
	};

	//+++ 3 s 0
	//++- ^ ^ 1
	//+-+ ^ s 2
	//+-- ^ ^ 3
	//-++ 7 s 4
	//-+- ^ ^ 5
	//--+ ^ s 6
	//--- ^ ^ 7

	stage1(3, 1);
	stage2(3, 1);
	stage3(0, 1);         // write
	stage3(1, -1);        // write
	stage2(3, -1);
	stage3(2, 1);         // write
	stage3(3, -1);        // write

	stage1(7, -1);
	stage2(7, 1);
	stage3(4, 1);         // write
	stage3(5, -1);        // write
	stage2(7, -1);
	stage3(6, 1);         // write
	stage3(7, -1);        // write

	//for (int i = 0; i < 8; ++i)
	//{
	//	auto &distance_map_i = volume.get_distance_map(i);

	//	// Bind images
	//	command_buffer.bind_input(*distance_map_i.image_view, 0, 0, 0);
	//	command_buffer.bind_input(*occupancy_map.image_view, 0, 1, 0);

	//	// Assign direction
	//	glm::vec3 direction((i / 1) % 2 == 0 ? 1.0f : -1.0f,
	//	                    (i / 2) % 2 == 0 ? 1.0f : -1.0f,
	//	                    (i / 4) % 2 == 0 ? 1.0f : -1.0f);
	//	command_buffer.push_constants<glm::vec3>(sizeof(glm::vec4), direction);

	//	// Dispatch 1st stage
	//	command_buffer.push_constants<uint32_t>(0, 0);
	//	command_buffer.dispatch(rndUp(extent.height, 8), rndUp(extent.depth, 8), 1);
	//	command_buffer.image_memory_barrier(*distance_map_i.image_view, memory_barrier_write_to_read);

	//	// Dispatch 2nd stage
	//	command_buffer.bind_input(*swap.image_view, 0, 1, 0);
	//	command_buffer.push_constants<uint32_t>(0, 1);
	//	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.depth, 8), 1);
	//	command_buffer.image_memory_barrier(*swap.image_view, memory_barrier_write_to_read);

	//	// Dispatch 3rd stage
	//	command_buffer.push_constants<uint32_t>(0, 2);
	//	command_buffer.dispatch(rndUp(extent.width, 8), rndUp(extent.height, 8), 1);
	//	command_buffer.image_memory_barrier(*distance_map_i.image_view, memory_barrier_write_to_read);
	//}

	for (int i = 0; i < 8; ++i)
	{
		auto &distance = volume.get_distance_map(i);
		command_buffer.image_memory_barrier(*distance.image_view, memory_barrier_compute_to_fragment);
	}
}
