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

#include "core/buffer.h"
#include "core/command_buffer.h"
#include "rendering/render_context.h"
#include "rendering/subpass.h"
#include "scene_graph/components/camera.h"
#include "scene_graph/scene.h"

#include "volume_component.h"

///**
//* @brief Uniform stucture for volume rendering
//* Camera position in texture coordinates is used to find the ray intersection with the back of the cube.
//* model_inv and plane used for box/plane intersection.
//*/
struct VolumeRenderUniform
{
	glm::mat4 model;                          // model matrix
	glm::mat4 model_inv;                      // model matrix (inverse)
	glm::mat4 camera_view;                    // camera view matrix
	glm::mat4 camera_proj;                    // camera projection matrix
	glm::mat4 camera_view_proj_inv;           // camera view projection matrix (inverse)
	glm::vec4 plane;                          // plane in global coordinates
	glm::vec4 plane_tex;                      // plane in texture coordinates
	glm::vec4 camera_pos_tex;                 // camera position in texture coordinates
	int       front_index;                    // index of the front vertex on the cube (see volume_render_plane_intersection.vert)
	float     sampling_factor;                // controls the voxel sampling density
	float     voxel_alpha_factor;             // all voxel alpha values are multiplied by this factor
	float     grad_magnitude_modifier;        // all gradient magnitudes are multiplied by this factor (to compensate for low image gradients)

	// NOTE: Temporary parameters for 2D transfer function (in place of texture)
	float intensity_min;
	float intensity_max;
	float gradient_min;
	float gradient_max;
};

class VolumeRenderSubpass : public vkb::Subpass
{
  public:
	enum class SkippingType : int
	{
		Distance = 0,
		Block    = 1,
		None     = 2,
	};

	enum class Test : int
	{
		None              = 0,
		RayEntry          = 1,
		RayExit           = 2,
		NumTextureSamples = 3,
	};

	struct Options
	{
		SkippingType skipping_type         = SkippingType::Distance;
		float        clip_distance         = 50.0f;
		bool         early_ray_termination = true;
		bool         depth_attachment      = false;
		Test         test                  = Test::None;
	};

	VolumeRenderSubpass(vkb::RenderContext &render_context, vkb::sg::Scene &scene, vkb::sg::Camera &camera, Options options);
	virtual ~VolumeRenderSubpass() = default;

	void draw(vkb::CommandBuffer &command_buffer) override;

  private:
	vkb::ShaderSource     vertex_source_plane_intersection;
	vkb::sg::Camera &     camera;
	std::vector<Volume *> volumes;

	std::unique_ptr<vkb::core::Buffer> vertex_buffer, index_buffer, index_buffer_plane_intersection;
	uint32_t                           index_count, index_count_plane_intersection;

	// Options
	Options            options;
	vkb::ShaderVariant shader_variant{};
};
