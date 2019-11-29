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

#include "transfer_function.h"

class Volume : public vkb::sg::Component
{
  public:
	Volume(const std::string &name);
	virtual ~Volume() = default;

	bool load_from_file(vkb::RenderContext &render_context, std::string filename, uint32_t distance_map_block_size = 4);

	void set_image_transform(const glm::mat4 &mat);

	void set_number_of_distance_maps(vkb::RenderContext &render_context, size_t n);

	virtual std::type_index get_type() override;

	struct Options
	{
		float sampling_factor          = 1.0f;
		float voxel_alpha_factor       = 1.0f;
		bool  use_precomputed_gradient = true;

		// Parameters defining simple grayscale 2D transfer function
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

	const Image &get_volume() const;
	const Image &get_gradient() const;
	const Image &get_transfer_function() const;
	const Image &get_distance_map(size_t idx = 0) const;
	const Image &get_distance_map_swap() const;

	glm::mat4 &get_image_transform();

	TransferFunctionUniform get_transfer_function_uniform();

	void update_transfer_function_texture(vkb::CommandBuffer &command_buffer);

	void           set_node(vkb::sg::Node &node);
	vkb::sg::Node *get_node() const;

	void upload_texture_with_staging(vkb::CommandBuffer &    command_buffer,
	                                 vkb::core::Buffer &     stage_buffer,
	                                 const vkb::core::Image &image, const vkb::core::ImageView &image_view);

  private:
	vkb::sg::Node *node;

	Image                              volume, gradient, transfer_function;
	std::unique_ptr<vkb::core::Buffer> transfer_function_staging;
	std::vector<Image>                 distance_maps;
	Image                              distance_map_swap;

	glm::mat4 image_transform;
};
