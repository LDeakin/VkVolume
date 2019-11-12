#version 320 es
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

precision highp float;

layout(set = 0, binding = 1, std140) uniform VolumeRenderUniform {
    mat4 model;
    mat4 model_inv;
    mat4 view;
    mat4 proj;
    mat4 view_proj_inv;
    vec4 plane;
    vec4 plane_tex;
    vec4 cam_pos_tex;
    int front_index;
    float sampling_factor;
    float voxel_alpha_factor;
    float value_min;
    float value_max;
} volume_uniform;

layout(location = 0) out vec4 pos_view_space;
layout(location = 1) out vec3 ray_entry;

// A Vertex Program for Efficient Box-Plane Intersection
// Christof Rezk Salama and Andreas Kolb, 2005

// Coordinate system
//         x
//        /
//  z -- O
//       |
//       y

//    Standard cube   ->       Authors layout (Salama&Kolb)
//      5 ------- 1            4 ------- 1
//     /         /|           /         /|
//    /         / |          /         / |
//   /         /  |         /         /  |
//  4 ------- 0   |        3 ------- 0   |  <-- closest = 0
//  |   7 ----|-- 3   ->   |   7 ----|-- 5
//  |  /      |  /         |  /      |  /
//  | /       | /          | /       | /
//  |/        |/           |/        |/
//  6 ------- 2            6 ------- 2

// Vertices of the authors cube layout on a unit cube, comments show standard cube layout
vec4 cube_vertices[8] = vec4[](
    vec4(0.0f, 0.0f, 0.0f, 1), // 0
    vec4(1.0f, 0.0f, 0.0f, 1), // 1
    vec4(0.0f, 1.0f, 0.0f, 1), // 2
    vec4(0.0f, 0.0f, 1.0f, 1), // 4
    vec4(1.0f, 0.0f, 1.0f, 1), // 5
    vec4(1.0f, 1.0f, 0.0f, 1), // 3
    vec4(0.0f, 1.0f, 1.0f, 1), // 6
    vec4(1.0f, 1.0f, 1.0f, 1)  // 7
);

// The mapping from standard cube to authors layout
const int index_map_standard_to_author[] = int[](0, 1, 2, 5, 3, 4, 6, 7);

// This holds the "vertex sequence" which varies depending on which vertex is furthest behind the clipping plane.
// It was written by swapping the 0th and nth vertices, and looking at where all vertices are moved.
// The ambigous swap for vertex 7 has vertex 4 at +x. This is all done with the authors cube layout.
int vertex_sequence[8*8] = int[](
    0, 1, 2, 3, 4, 5, 6, 7,
    1, 0, 4, 5, 2, 3, 7, 6,
    2, 6, 0, 5, 7, 3, 1, 4,
    3, 6, 4, 0, 2, 7, 1, 5,
    4, 3, 7, 1, 0, 6, 5, 2,
    5, 2, 1, 7, 6, 0, 4, 3,
    6, 7, 3, 2, 5, 4, 0, 1,
    7, 4, 6, 5, 1, 3, 2, 0
);

// This defines the intersection tests P0 -> P6 outlined in section 4 of (Salama & Kolb, 2005) 
int _V[6 * 4 * 2] = int[](
    0, 1, 1, 4, 4, 7, 4, 7, // P0
    1, 5, 0, 1, 1, 4, 4, 7, // P1
    0, 2, 2, 5, 5, 7, 5, 7, // P2
    2, 6, 0, 2, 2, 5, 5, 7, // P3
    0, 3, 3, 6, 6, 7, 6, 7, // P4
    3, 4, 0, 3, 3, 6, 6, 7  // P5
);

void main()
{
    int sequence_index = index_map_standard_to_author[volume_uniform.front_index] * 8;
    vec3 pos_tex = vec3(1.0 / 0.0f); // set to nan by default
    for (int e = 0; e < 4; ++e)
    {
        int vidx1 = vertex_sequence[sequence_index + _V[(int(gl_VertexIndex) * 4 + e) * 2]];
        int vidx2 = vertex_sequence[sequence_index + _V[(int(gl_VertexIndex) * 4 + e) * 2 + 1]];

        vec3 vecV1 = vec3(cube_vertices[vidx1]);
        vec3 vecV2 = vec3(cube_vertices[vidx2]);
        vec3 vecDir = vecV2 - vecV1;

        float denom = dot(vecDir, volume_uniform.plane_tex.xyz);
        float lambda = denom != 0.0f ? (-volume_uniform.plane_tex.w - dot(vecV1, volume_uniform.plane_tex.xyz)) / denom : -1.0;

        if (lambda >= 0.0f && lambda <= 1.0f)
        {
            pos_tex = vecV1 + lambda * vecDir;
            break;
        }
    }

    pos_view_space = volume_uniform.view * volume_uniform.model * vec4(pos_tex - 0.5f, 1.0f);
    ray_entry = pos_tex;
    gl_Position = volume_uniform.proj * pos_view_space;
}
