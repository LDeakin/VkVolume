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
#include <glm/gtc/quaternion.hpp>

#include "scene_graph/scripts/free_camera.h"

struct OrbitCamera : public vkb::sg::FreeCamera
{
	OrbitCamera(vkb::sg::Node &node);
	virtual ~OrbitCamera() = default;

	virtual void update(float delta_time) override;

	void recalculate_view();

	// Variables
	glm::vec3 position_ = glm::vec3(0, 0, 0);
	glm::quat rotation_ = glm::quat(1, 0, 0, 0);
	float     zoom_     = -100.0f;
};
