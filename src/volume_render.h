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

#include "vulkan_sample.h"

#include "scene_graph/components/camera.h"

#include "compute_distance_map.h"
#include "volume_render_subpass.h"

class VolumeRender : public vkb::VulkanSample
{
  public:
	VolumeRender();
	virtual ~VolumeRender() = default;

	virtual bool prepare(vkb::Platform &platform) override;

	virtual void update(float delta_time) override;

	virtual void get_device_features() override;

  private:
	virtual void      prepare_render_context() override;
	vkb::RenderTarget create_render_target(vkb::core::Image &&swapchain_image);

	vkb::sg::Node &add_orbit_camera(const std::string &node_name);
	void           update_distance_map(Volume &volume);
	void           init_render_pipeline();

	virtual void draw_gui() override;

	vkb::sg::Camera *camera;

	std::unique_ptr<ComputeDistanceMap> compute_distance_map;

	// Options
	int                          block_size;
	VolumeRenderSubpass::Options volume_render_options;
	bool                         render_sponza_scene;
	bool                         spin_volumes;
};

std::unique_ptr<vkb::VulkanSample> create_volume_render();
