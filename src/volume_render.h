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
#include "compute_gradient_map.h"
#include "compute_occupied_voxel_count.h"
#include "volume_render_subpass.h"

#include "platform/plugins/plugin_base.h"

class VolumeRenderPlugin;

using VolumeRenderPluginTags = vkb::PluginBase<VolumeRenderPlugin, vkb::tags::Passive>;

class VolumeRenderPlugin : public VolumeRenderPluginTags
{
  public:
	VolumeRenderPlugin();

	virtual ~VolumeRenderPlugin() = default;

	virtual bool is_active(const vkb::CommandParser &parser) override;

	virtual void init(const vkb::CommandParser &parser) override;

	vkb::FlagCommand imin_flag{vkb::FlagType::OneValue, "imin", "", "Intensity minimum"};
	vkb::FlagCommand imax_flag{vkb::FlagType::OneValue, "imax", "", "Intensity maximum"};
	vkb::FlagCommand gmin_flag{vkb::FlagType::OneValue, "gmin", "", "Gradient minimum"};
	vkb::FlagCommand gmax_flag{vkb::FlagType::OneValue, "gmax", "", "Gradient maximum"};
	vkb::FlagCommand skipmode_flag{vkb::FlagType::OneValue, "skipmode", "", "Skipping mode 0=None, 1=Block 2=Distance 3=DistanceAnisotropic"};
	vkb::FlagCommand blocksize_flag{vkb::FlagType::OneValue, "blocksize", "", "Block size edge length"};
	vkb::FlagCommand gradient_test_flag{vkb::FlagType::FlagOnly, "gradient_test", "", "Gradient test"};
	//vkb::FlagCommand datasets_flag{vkb::FlagType::ManyValues, "datasets", "D", "Dataset filesnames"};
	vkb::PositionalCommand dataset_flag{"dataset", "Dataset filename"};

	vkb::CommandGroup cmd{"Volume Render Options", {&imin_flag, &imax_flag, &gmin_flag, &gmax_flag, &skipmode_flag, &blocksize_flag, &gradient_test_flag, &dataset_flag}};

	float                             imin, imax, gmin, gmax;
	VolumeRenderSubpass::SkippingType skipmode;
	int                               blocksize;
	bool                              gradient_test;
	std::vector<std::string>          datasets;
};

class VolumeRender : public vkb::VulkanSample
{
  public:
	VolumeRender();
	virtual ~VolumeRender() = default;

	virtual bool prepare(vkb::Platform &platform) override;

	virtual void update(float delta_time) override;

	virtual void request_gpu_features(vkb::PhysicalDevice &gpu) override;

  private:
	virtual void                       prepare_render_context() override;
	std::unique_ptr<vkb::RenderTarget> create_render_target(vkb::core::Image &&swapchain_image);

	void VolumeRender::update_transfer_function(Volume &volume);

	vkb::CommandBuffer &compute_start();
	void                compute_submit(vkb::CommandBuffer &command_buffer);

	void init_render_pipeline();

	virtual void draw_gui() override;

	vkb::sg::Camera *camera;

	std::unique_ptr<ComputeDistanceMap>        compute_distance_map;
	std::unique_ptr<ComputeGradientMap>        compute_gradient_map;
	std::unique_ptr<ComputeOccupiedVoxelCount> compute_occupied_voxel_count;

	// Options
	VolumeRenderSubpass::Options volume_render_options;
	bool                         render_sponza_scene;
	bool                         spin_volumes;
};

std::unique_ptr<vkb::VulkanSample> create_volume_render();
