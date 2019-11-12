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

#include "volume_component.h"

#include "load_volume.h"

using namespace vkb;

Volume::Volume(const std::string &name) :
    Component{name},
    image_transform(glm::mat4(1.0f))
{}

core::Buffer Volume::upload_texture_with_staging(CommandBuffer &command_buffer, const uint8_t *data, VkDeviceSize data_size, const core::Image &image, const core::ImageView &image_view)
{
	// Upload data into the vulkan image memory
	core::Buffer stage_buffer{command_buffer.get_device(), data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, 0};
	stage_buffer.update({data, data + data_size});

	{
		// Prepare for transfer
		ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_HOST_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;

		command_buffer.image_memory_barrier(image_view, memory_barrier);
	}

	// Copy
	VkBufferImageCopy buffer_copy_region{};
	buffer_copy_region.imageSubresource.layerCount = image_view.get_subresource_range().layerCount;
	buffer_copy_region.imageSubresource.aspectMask = image_view.get_subresource_range().aspectMask;
	buffer_copy_region.imageExtent                 = image.get_extent();

	command_buffer.copy_buffer_to_image(stage_buffer, image, {buffer_copy_region});

	return stage_buffer;
}

bool Volume::load_from_file(vkb::RenderContext &render_context, std::string filename, uint32_t distance_map_block_size /* = 4 */)
{
	using namespace vkb;

	auto                 header      = LoadVolume::load_header(filename + ".header");
	std::vector<uint8_t> volume_data = LoadVolume::load_data(filename, header);
	size_t               data_size   = volume_data.size() * sizeof(uint8_t);
	auto &               extent      = header.extent;
	set_image_transform(header.image_transform);

	auto &device = render_context.get_device();

	// Create volume image and upload
	volume.image      = std::make_unique<core::Image>(device, extent, VK_FORMAT_R8_UNORM,
                                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                 VMA_MEMORY_USAGE_GPU_ONLY);
	volume.image_view = std::make_unique<core::ImageView>(*volume.image, VK_IMAGE_VIEW_TYPE_3D);

	// Create occupancy map and distance map (populated later with compute shader)
	VkExtent3D extent_occupancy  = {extent.width / distance_map_block_size, extent.height / distance_map_block_size, extent.depth / distance_map_block_size};
	distance_map.image           = std::make_unique<core::Image>(device, extent_occupancy, VK_FORMAT_R8_UINT,
                                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                       VMA_MEMORY_USAGE_GPU_ONLY);
	distance_map.image_view      = std::make_unique<core::ImageView>(*distance_map.image, VK_IMAGE_VIEW_TYPE_3D);
	distance_map_swap.image      = std::make_unique<core::Image>(device, extent_occupancy, VK_FORMAT_R8_UINT,
                                                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                            VMA_MEMORY_USAGE_GPU_ONLY);
	distance_map_swap.image_view = std::make_unique<core::ImageView>(*distance_map_swap.image, VK_IMAGE_VIEW_TYPE_3D);

	// Begin recording
	{
		auto &    command_buffer = device.request_command_buffer();
		FencePool fence_pool{device};
		command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

		auto staging = upload_texture_with_staging(command_buffer, volume_data.data(), data_size, *volume.image, *volume.image_view);

		{
			// Prepare volume for fragment shader
			ImageMemoryBarrier memory_barrier{};
			memory_barrier.old_layout      = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			memory_barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			memory_barrier.src_access_mask = 0;
			memory_barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
			memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
			memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

			command_buffer.image_memory_barrier(*volume.image_view, memory_barrier);
		}

		{
			// Prepare occupancy and distance map for compute
			vkb::ImageMemoryBarrier memory_barrier{};
			memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
			memory_barrier.new_layout      = VK_IMAGE_LAYOUT_GENERAL;
			memory_barrier.src_access_mask = 0;
			memory_barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
			memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
			memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			command_buffer.image_memory_barrier(*distance_map_swap.image_view, memory_barrier);
			memory_barrier.new_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			command_buffer.image_memory_barrier(*distance_map.image_view, memory_barrier);
		}

		// End recording
		command_buffer.end();
		auto &queue = device.get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT, 0);
		queue.submit(command_buffer, device.request_fence());

		// Wait for the command buffer to finish its work before destroying the staging buffer
		device.get_fence_pool().wait();
		device.get_fence_pool().reset();
		device.get_command_pool().reset_pool();
	}

	// Create volume sampler
	VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	sampler_info.maxAnisotropy = 1.0f;
	sampler_info.magFilter     = VK_FILTER_LINEAR;
	sampler_info.minFilter     = VK_FILTER_LINEAR;
	sampler_info.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_info.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	volume.sampler             = std::make_unique<core::Sampler>(device, sampler_info);

	// Create distance map sampler
	sampler_info.magFilter = VK_FILTER_NEAREST;
	sampler_info.minFilter = VK_FILTER_NEAREST;
	distance_map.sampler   = std::make_unique<core::Sampler>(device, sampler_info);

	return true;
}

void Volume::set_image_transform(const glm::mat4 &mat)
{
	image_transform = mat;
}

std::type_index Volume::get_type()
{
	return typeid(Volume);
}

const Volume::Image &Volume::get_volume()
{
	return volume;
}

const Volume::Image &Volume::get_distance_map()
{
	return distance_map;
}

const Volume::Image &Volume::get_distance_map_swap()
{
	return distance_map_swap;
}

glm::mat4 &Volume::get_image_transform()
{
	return image_transform;
}

void Volume::set_node(vkb::sg::Node &node)
{
	this->node = &node;
}

vkb::sg::Node *Volume::get_node() const
{
	return node;
}
