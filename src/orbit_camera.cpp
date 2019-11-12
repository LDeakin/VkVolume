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

#include "orbit_camera.h"

#include "scene_graph/components/transform.h"
#include "scene_graph/node.h"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

using namespace vkb;

OrbitCamera::OrbitCamera(sg::Node &node) :
    FreeCamera(node)
{
	recalculate_view();
}

void OrbitCamera::recalculate_view()
{
	const glm::mat4 view = glm::translate(glm::vec3(0, 0, zoom_)) *
	                       glm::inverse(glm::translate(position_) * glm::toMat4(rotation_));

	auto &transform = get_node().get_component<sg::Transform>();
	transform.set_matrix(glm::inverse(view));
}

void OrbitCamera::update(float delta_time)
{
	const float     zoomSpeedWheel = 0.02f;
	const float     zoomSpeedDrag  = 0.2f;
	const float     panSpeed       = 0.2f;
	constexpr float rotationSpeed  = glm::radians(1.0f);        // 1 pixel = 1 degree

	bool changed = false;

	//// Zoom
	//if (mouseWheelDelta != 0.0f)
	//{
	//	zoom_ += mouseWheelDelta * zoomSpeedWheel;
	//	changed = true;
	//}

	// Zoom2
	if (mouse_button_pressed[MouseButton::Right] && mouse_move_delta.y != 0.0f)
	{
		zoom_ += -mouse_move_delta.y * zoomSpeedDrag;
		changed = true;
	}

	// Rotate
	if (mouse_button_pressed[MouseButton::Left] && (mouse_move_delta.x != 0.0f || mouse_move_delta.y != 0.f))
	{
		rotation_ = glm::angleAxis(-mouse_move_delta.x * rotationSpeed, glm::vec3(glm::rotate(rotation_, glm::vec4(0, 1, 0, 0)))) * rotation_;
		rotation_ = glm::angleAxis(-mouse_move_delta.y * rotationSpeed, glm::vec3(glm::rotate(rotation_, glm::vec4(1, 0, 0, 0)))) * rotation_;
		rotation_ = glm::normalize(rotation_);
		changed   = true;
	}

	// Pan
	if (mouse_button_pressed[MouseButton::Middle] && (mouse_move_delta.x != 0.0f || mouse_move_delta.y != 0.f))
	{
		changed = true;
		position_ += glm::vec3(glm::rotate(rotation_, glm::vec4(-mouse_move_delta.x * panSpeed, mouse_move_delta.y * panSpeed, 0, 0)));
	}

	if (changed)
	{
		recalculate_view();
	}

	mouse_move_delta = {};
	touch_move_delta = {};
}
