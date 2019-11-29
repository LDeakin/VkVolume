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

void Volume::upload_texture_with_staging(CommandBuffer &    command_buffer,
                                         vkb::core::Buffer &stage_buffer,
                                         const core::Image &image, const core::ImageView &image_view)
{
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

	// Create transfer function
	VkExtent3D tf_extent         = {256, 256, 1};
	transfer_function.image      = std::make_unique<core::Image>(device, tf_extent, VK_FORMAT_R8G8B8A8_UNORM,
                                                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                            VMA_MEMORY_USAGE_GPU_ONLY);
	transfer_function.image_view = std::make_unique<core::ImageView>(*transfer_function.image, VK_IMAGE_VIEW_TYPE_2D);
	transfer_function_staging    = std::make_unique<core::Buffer>(device, 256 * 256 * sizeof(glm::u8vec4), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, 0);

	// Create volume image and upload
	volume.image      = std::make_unique<core::Image>(device, extent, VK_FORMAT_R8_UNORM,
                                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                 VMA_MEMORY_USAGE_GPU_ONLY);
	volume.image_view = std::make_unique<core::ImageView>(*volume.image, VK_IMAGE_VIEW_TYPE_3D);

	if (options.use_precomputed_gradient)
	{
		gradient.image      = std::make_unique<core::Image>(device, extent, VK_FORMAT_R8_UNORM,
                                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                       VMA_MEMORY_USAGE_GPU_ONLY);
		gradient.image_view = std::make_unique<core::ImageView>(*gradient.image, VK_IMAGE_VIEW_TYPE_3D);
	}

	// Create swap image (populated later with compute shader)
	// Distance maps are created with a call to set_number_of_distance_maps()
	auto       rndUp             = [](uint32_t x, uint32_t y) { return (x + y - 1) / y; };
	VkExtent3D extent_occupancy  = {rndUp(extent.width, distance_map_block_size), rndUp(extent.height, distance_map_block_size), rndUp(extent.depth, distance_map_block_size)};
	distance_map_swap.image      = std::make_unique<core::Image>(device, extent_occupancy, VK_FORMAT_R8_UINT,
                                                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                            VMA_MEMORY_USAGE_GPU_ONLY);
	distance_map_swap.image_view = std::make_unique<core::ImageView>(*distance_map_swap.image, VK_IMAGE_VIEW_TYPE_3D);

	// Upload volume image
	{
		auto &    command_buffer = device.request_command_buffer();
		FencePool fence_pool{device};
		command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

		// Upload data into the vulkan image memory
		core::Buffer stage_buffer{command_buffer.get_device(), data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY, 0};
		stage_buffer.update({volume_data.data(), volume_data.data() + data_size});
		upload_texture_with_staging(command_buffer, stage_buffer, *volume.image, *volume.image_view);

		{
			// Prepare volume and gradient for fragment shader
			ImageMemoryBarrier memory_barrier{};
			memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
			memory_barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			memory_barrier.src_access_mask = 0;
			memory_barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
			memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
			memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

			command_buffer.image_memory_barrier(*volume.image_view, memory_barrier);
			if (options.use_precomputed_gradient)
			{
				command_buffer.image_memory_barrier(*gradient.image_view, memory_barrier);
			}
			command_buffer.image_memory_barrier(*transfer_function.image_view, memory_barrier);
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

	// Create samplers
	VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	sampler_info.maxAnisotropy = 1.0f;
	sampler_info.magFilter     = VK_FILTER_LINEAR;
	sampler_info.minFilter     = VK_FILTER_LINEAR;
	sampler_info.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_info.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	volume.sampler             = std::make_unique<core::Sampler>(device, sampler_info);
	gradient.sampler           = std::make_unique<core::Sampler>(device, sampler_info);
	sampler_info.magFilter     = VK_FILTER_NEAREST;
	sampler_info.minFilter     = VK_FILTER_NEAREST;
	transfer_function.sampler  = std::make_unique<core::Sampler>(device, sampler_info);
	return true;
}

void Volume::set_number_of_distance_maps(vkb::RenderContext &render_context, size_t n)
{
	if (n <= distance_maps.size())
	{
		return;
	}

	distance_maps.resize(n);

	auto &device = render_context.get_device();

	// Create samplers
	VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	sampler_info.maxAnisotropy = 1.0f;
	sampler_info.magFilter     = VK_FILTER_NEAREST;
	sampler_info.minFilter     = VK_FILTER_NEAREST;
	sampler_info.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_info.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_info.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	for (auto &distance_map : distance_maps)
	{
		distance_map.image      = std::make_unique<core::Image>(device, distance_map_swap.image->get_extent(), distance_map_swap.image->get_format(),
                                                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                                           VMA_MEMORY_USAGE_GPU_ONLY);
		distance_map.image_view = std::make_unique<core::ImageView>(*distance_map.image, VK_IMAGE_VIEW_TYPE_3D);
		distance_map.sampler    = std::make_unique<core::Sampler>(device, sampler_info);
	}
}

void Volume::set_image_transform(const glm::mat4 &mat)
{
	image_transform = mat;
}

std::type_index Volume::get_type()
{
	return typeid(Volume);
}

const Volume::Image &Volume::get_volume() const
{
	return volume;
}

const Volume::Image &Volume::get_gradient() const
{
	return gradient;
}

const Volume::Image &Volume::get_transfer_function() const
{
	return transfer_function;
}

const Volume::Image &Volume::get_distance_map(size_t idx /* = 0 */) const
{
	return distance_maps.at(idx);
}

const Volume::Image &Volume::get_distance_map_swap() const
{
	return distance_map_swap;
}

glm::mat4 &Volume::get_image_transform()
{
	return image_transform;
}

TransferFunctionUniform Volume::get_transfer_function_uniform()
{
	TransferFunctionUniform transfer_function_uniform;
	transfer_function_uniform.sampling_factor         = options.sampling_factor;
	transfer_function_uniform.voxel_alpha_factor      = options.voxel_alpha_factor;
	transfer_function_uniform.grad_magnitude_modifier = 1.0f;
	transfer_function_uniform.use_gradient            = options.gradient_max != options.gradient_min;
#ifndef TRANSFER_FUNCTION_TEXTURE
	transfer_function_uniform.intensity_min       = options.intensity_min;
	transfer_function_uniform.intensity_range_inv = 1.0f / (options.intensity_max - options.intensity_min);
	transfer_function_uniform.gradient_min        = options.gradient_min;
	transfer_function_uniform.gradient_range_inv  = 1.0f / (options.gradient_max - options.gradient_min);
#endif
	return transfer_function_uniform;
}

void Volume::update_transfer_function_texture(vkb::CommandBuffer &command_buffer)
{
	// Update the transfer function texture

	std::vector<glm::u8vec4> tex(256 * 256);        // FIXME: Allocate once
	auto                     clamp = [](float x, float min, float max) {
        return std::min(std::max(x, min), max);
	};
	float  i_inv        = 1.0f / (options.intensity_max - options.intensity_min);
	float  g_inv        = 1.0f / (options.gradient_max - options.gradient_min);
	bool   use_gradient = options.gradient_max != options.gradient_min;
	size_t idx          = 0;
	for (float g = 0; g < 256; ++g)
		for (float i = 0; i < 256; ++i, idx++)
		{
			float   alpha_i = clamp(((i / 255.0f) - options.intensity_min) * i_inv, 0.0f, 1.0f);
			float   alpha_g = use_gradient ? clamp(((g / 255.0f) - options.gradient_min) * g_inv, 0.0f, 1.0f) : 1.0f;
			uint8_t alpha   = static_cast<uint8_t>(clamp(alpha_i * alpha_g * 255, 0, 255));
			tex.at(idx)     = glm::u8vec4(alpha);
		}
	transfer_function_staging->update(reinterpret_cast<const uint8_t *>(tex.data()), tex.size() * sizeof(glm::u8vec4));
	upload_texture_with_staging(command_buffer, *transfer_function_staging,
	                            *get_transfer_function().image, *get_transfer_function().image_view);

	{
		// Prepare transfer function for fragment shader
		ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TRANSFER_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		command_buffer.image_memory_barrier(*get_transfer_function().image_view, memory_barrier);
	}
}

void Volume::set_node(vkb::sg::Node &node)
{
	this->node = &node;
}

vkb::sg::Node *Volume::get_node() const
{
	return node;
}
