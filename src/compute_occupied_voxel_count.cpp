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

#include "compute_occupied_voxel_count.h"

#include "common/vk_common.h"
#include "platform/filesystem.h"
#include "rendering/render_context.h"

#include "transfer_function.h"

auto rndUp = [](int x, int y) { return (x + y - 1) / y; };

ComputeOccupiedVoxelCount::ComputeOccupiedVoxelCount(vkb::RenderContext &render_context) :
    render_context(render_context),
    compute_shader(vkb::fs::read_shader("occupied_voxel_count.comp")),
    compute_shader_reduce(vkb::fs::read_shader("occupied_voxel_count_reduce.comp"))
{
	// Get subgroup size
	render_context.get_device().get_features();
	VkPhysicalDeviceSubgroupProperties subgroup_properties{};
	subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	VkPhysicalDeviceProperties2 properties{};
	properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	properties.pNext = &subgroup_properties;
	vkGetPhysicalDeviceProperties2(render_context.get_device().get_physical_device(), &properties);
	subgroup_size = subgroup_properties.subgroupSize;

	// Build all shaders upfront
	vkb::ShaderVariant variant;
	variant.add_define("PRECOMPUTED_GRADIENT");
	auto &resource_cache = render_context.get_device().get_resource_cache();
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader);
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader, variant);

	variant_reduce.add_define("SUBGROUP_SIZE " + std::to_string(subgroup_size));
	resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_reduce, variant_reduce);

	memory_barrier_compute.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	memory_barrier_compute.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
	memory_barrier_compute.src_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
	memory_barrier_compute.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_compute.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	memory_barrier_compute.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

	memory_barrier_shader_read_only_optimal.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
	memory_barrier_shader_read_only_optimal.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	memory_barrier_shader_read_only_optimal.src_access_mask = VK_ACCESS_SHADER_WRITE_BIT;
	memory_barrier_shader_read_only_optimal.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
	memory_barrier_shader_read_only_optimal.src_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	memory_barrier_shader_read_only_optimal.dst_stage_mask  = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
}

vkb::core::Buffer ComputeOccupiedVoxelCount::initialise_buffer(vkb::Device &device, Volume &volume)
{
	const VkExtent3D   extent = volume.get_volume().image->get_extent();
	const glm::uvec3   dispatchSize(rndUp(extent.width, 8), rndUp(extent.height, 8), rndUp(extent.depth, 8));
	const VkDeviceSize nElements =
	    static_cast<VkDeviceSize>(dispatchSize.x) * static_cast<VkDeviceSize>(dispatchSize.y) *
	    static_cast<VkDeviceSize>(dispatchSize.z) * static_cast<VkDeviceSize>(8 * 8 * 8 / subgroup_size);
	const VkDeviceSize bufferSize = sizeof(uint64_t) * nElements;
	vkb::core::Buffer  buffer(device, bufferSize,
                             VkBufferUsageFlagBits::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_TO_CPU);
	return buffer;
}

void ComputeOccupiedVoxelCount::compute(vkb::CommandBuffer &command_buffer, Volume &volume, vkb::BufferAllocation &buffer, vkb::BufferAllocation &transfer_function_uniform)
{
	auto &resource_cache = command_buffer.get_device().get_resource_cache();

	// Set layout
	command_buffer.image_memory_barrier(*volume.get_volume().image_view, memory_barrier_compute);

	// Run the count
	{
		vkb::ShaderVariant variant;
		if (volume.options.use_precomputed_gradient)
		{
			variant.add_define("PRECOMPUTED_GRADIENT");
		}

		auto &shader_module = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader, variant);
		shader_module.set_resource_dynamic("countBuffer");
		auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});
		command_buffer.bind_pipeline_layout(pipeline_layout);
		command_buffer.bind_input(*volume.get_volume().image_view, 0, 0, 0);
		command_buffer.bind_buffer(transfer_function_uniform.get_buffer(), transfer_function_uniform.get_offset(), transfer_function_uniform.get_size(), 0, 1, 0);
		// transfer function texture
		if (volume.options.use_precomputed_gradient)
		{
			command_buffer.image_memory_barrier(*volume.get_gradient().image_view, memory_barrier_compute);
			command_buffer.bind_input(*volume.get_gradient().image_view, 0, 3, 0);
		}
		command_buffer.bind_buffer(buffer.get_buffer(), buffer.get_offset(), buffer.get_size(), 0, 4, 0);
		const VkExtent3D extent = volume.get_volume().image->get_extent();
		const glm::uvec3 dispatchSize(rndUp(extent.width, 8), rndUp(extent.height, 8), rndUp(extent.depth, 8));
		command_buffer.dispatch(dispatchSize.x, dispatchSize.y, dispatchSize.z);
		command_buffer.image_memory_barrier(*volume.get_volume().image_view, memory_barrier_shader_read_only_optimal);
		if (volume.options.use_precomputed_gradient)
		{
			command_buffer.image_memory_barrier(*volume.get_gradient().image_view, memory_barrier_shader_read_only_optimal);
		}
	}

	// Run the reduce operations
	{
		auto &shader_module   = resource_cache.request_shader_module(VK_SHADER_STAGE_COMPUTE_BIT, compute_shader_reduce, variant_reduce);
		auto &pipeline_layout = resource_cache.request_pipeline_layout({&shader_module});
		command_buffer.bind_pipeline_layout(pipeline_layout);
		command_buffer.bind_buffer(buffer.get_buffer(), buffer.get_offset(), buffer.get_size(), 0, 0, 0);

		uint32_t                 stride        = 1;
		vkb::BufferMemoryBarrier barrier;
		barrier.src_access_mask = barrier.dst_access_mask =
		    VkAccessFlagBits::VK_ACCESS_SHADER_READ_BIT | VkAccessFlagBits::VK_ACCESS_SHADER_WRITE_BIT;
		barrier.src_stage_mask = barrier.dst_stage_mask =
		    VkPipelineStageFlagBits::VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		auto         rndUp64   = [](uint64_t x, uint64_t y) { return (x + y - 1) / y; };
		const size_t nElements = buffer.get_size() / sizeof(uint64_t);
		while (stride < nElements)
		{
			command_buffer.push_constants<uint64_t>(0, static_cast<uint64_t>(nElements));
			command_buffer.push_constants<uint32_t>(sizeof(uint64_t), stride);
			command_buffer.dispatch(static_cast<uint32_t>(rndUp64(nElements, static_cast<uint64_t>(subgroup_size) * static_cast<uint64_t>(stride))), 1, 1);
			command_buffer.buffer_memory_barrier(buffer.get_buffer(), buffer.get_offset(), buffer.get_size(), barrier);
			stride *= subgroup_size;
		}
	}
}

uint64_t ComputeOccupiedVoxelCount::get_result(vkb::BufferAllocation &buffer) const
{
	buffer.get_buffer().flush();        // FIXME: Needed?
	auto     data  = buffer.get_buffer().map() + buffer.get_offset();
	uint64_t count = reinterpret_cast<const uint64_t *>(data)[0];
	buffer.get_buffer().unmap();
	return count;
}
