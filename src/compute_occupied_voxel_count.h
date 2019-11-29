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
#include "volume_render_subpass.h"

namespace vkb
{
class RenderContext;
class CommandBuffer;
}        // namespace vkb

class ComputeOccupiedVoxelCount
{
  public:
	ComputeOccupiedVoxelCount(vkb::RenderContext &render_context);

	virtual ~ComputeOccupiedVoxelCount() = default;

	vkb::core::Buffer initialise_buffer(vkb::Device &device, Volume &volume);
	void              compute(vkb::CommandBuffer &command_buffer, Volume &volume, vkb::BufferAllocation &buffer, vkb::BufferAllocation &transfer_function_uniform);
	uint64_t          get_result(vkb::BufferAllocation &buffer) const;

  private:
	vkb::RenderContext &render_context;

	vkb::ShaderSource compute_shader, compute_shader_reduce;

	vkb::ImageMemoryBarrier memory_barrier_compute, memory_barrier_shader_read_only_optimal{};

	uint32_t           subgroup_size;
	vkb::ShaderVariant variant_reduce;
};