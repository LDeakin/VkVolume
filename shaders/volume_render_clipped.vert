#version 450
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

#extension GL_EXT_clip_cull_distance: enable

precision highp float;

layout(location = 0) in vec3 position;

layout(set = 0, binding = 1) uniform CameraUniform {
    mat4 view;
    mat4 proj;
    mat4 view_proj_inv;
    mat4 model;
    mat4 model_inv;
} camera_uniform;

layout(set = 0, binding = 2) uniform RayCastUniform {
    vec4 plane;
    vec4 plane_tex;
    vec4 cam_pos_tex;
    vec4 block_size;
    int front_index;
} ray_cast_uniform;

layout(location = 0) out vec4 position_out;
layout(location = 1) out vec3 ray_entry;

out gl_PerVertex 
{
    vec4 gl_Position;
    float gl_ClipDistance[1];
};

void main()
{
    // Convert to world space
    vec4 position_world = camera_uniform.model * vec4(position, 1.0f);

    // Distance to clip plane
    gl_ClipDistance[0] = dot(ray_cast_uniform.plane, position_world);

    // Ray entry (in texel coordinates)
    ray_entry = position + 0.5f;

    // Output (projection space)
    position_out = camera_uniform.proj * camera_uniform.view * position_world;
    gl_Position = position_out;

}
