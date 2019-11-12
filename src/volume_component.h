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

#pragma once

#include <glm/glm.hpp>

#include "core/image.h"
#include "core/image_view.h"
#include "core/sampler.h"
#include "rendering/render_context.h"
#include "scene_graph/component.h"
#include "scene_graph/node.h"

class Volume : public vkb::sg::Component
{
  public:
	Volume(const std::string &name);
	virtual ~Volume() = default;

	bool load_from_file(vkb::RenderContext &render_context, std::string filename, uint32_t distance_map_block_size = 4);

	void set_image_transform(const glm::mat4 &mat);

	virtual std::type_index get_type() override;

	struct Options
	{
		float sampling_factor    = 1.0f;
		float voxel_alpha_factor = 1.0f;

		// NOTE: Temporary parameters for a simple grayscale 2D transfer function (in place of texture-based lookup)
		float intensity_min = 0.0f;
		float intensity_max = 1.0f;
		float gradient_min  = 0.0f;
		float gradient_max  = 1.0f;
	} options;

	struct Image
	{
		std::unique_ptr<vkb::core::Image>     image;
		std::unique_ptr<vkb::core::ImageView> image_view;
		std::unique_ptr<vkb::core::Sampler>   sampler;
	};

	const Image &get_volume();
	//const Image &get_gradient();
	//const Image &get_transfer_function();
	const Image &get_distance_map();
	const Image &get_distance_map_swap();

	glm::mat4 &get_image_transform();

	void           set_node(vkb::sg::Node &node);
	vkb::sg::Node *get_node() const;

  private:
	vkb::sg::Node *node;

	Image volume, distance_map, distance_map_swap;

	glm::mat4 image_transform;

	vkb::core::Buffer upload_texture_with_staging(vkb::CommandBuffer &command_buffer, const uint8_t *data, VkDeviceSize data_size, const vkb::core::Image &image, const vkb::core::ImageView &image_view);
};
