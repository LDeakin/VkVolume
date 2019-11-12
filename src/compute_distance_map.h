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

#include "core/shader_module.h"

#include "volume_component.h"

namespace vkb
{
class RenderContext;
class CommandBuffer;
}        // namespace vkb

class ComputeDistanceMap
{
  public:
	ComputeDistanceMap(vkb::RenderContext &render_context);

	virtual ~ComputeDistanceMap() = default;

	void compute(vkb::CommandBuffer &command_buffer, const Volume::Image &volume, const Volume::Image &distance_map, const Volume::Image &distance_map_swap,
	             float intensity_min, float intensity_max, float gradient_min, float gradient_max);

  private:
	void computeOccupancy(vkb::CommandBuffer &command_buffer, const Volume::Image &volume, const Volume::Image &occupancy_map,
	                      float intensity_min, float intensity_max, float gradient_min, float gradient_max);
	void computeDistance(vkb::CommandBuffer &command_buffer, const Volume::Image &occupancy_and_distance_map, const Volume::Image &distance_map_swap);

	vkb::RenderContext &render_context;

	vkb::ShaderSource compute_shader_occupancy;
	vkb::ShaderSource compute_shader_distance;

	vkb::ImageMemoryBarrier memory_barrier_fragment_to_compute{};
	vkb::ImageMemoryBarrier memory_barrier_write_to_read{};
	vkb::ImageMemoryBarrier memory_barrier_compute_to_fragment{};
};