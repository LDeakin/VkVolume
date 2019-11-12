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

#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class LoadVolume
{
  public:
	struct Header
	{
		VkExtent3D  extent;
		glm::vec3   voxel_size;
		glm::vec2   normalisation_range;
		std::string type;
		std::string endianness;
		glm::mat4   image_transform;
		glm::vec2   tf_range;
		float       alpha_factor;
	};

	static Header               load_header(std::string filename_header);
	static std::vector<uint8_t> load_data(std::string filename_data, const Header &header);

  private:
	template <typename T>
	static std::vector<uint8_t> load_data_impl(std::string filename_data, const Header &header);
};
