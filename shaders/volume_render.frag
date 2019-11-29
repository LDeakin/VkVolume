#version 460
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

#extension GL_GOOGLE_include_directive : enable

precision highp float;

#ifdef DEPTH_ATTACHMENT
layout (input_attachment_index = 0, binding = 0) uniform subpassInput i_depth;
#endif

layout(location = 0) in vec4 pos_view_space;
layout(location = 1) in vec3 ray_entry;

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

layout (set = 0, binding = 5) uniform mediump sampler3D volume;

#define TRANSFER_FUNCTION_SET 0
#define TRANSFER_FUNCTION_BINDING_UNIFORM 3
#define TRANSFER_FUNCTION_BINDING_TEXTURE 4
#include "transfer_function.glsl"

#ifdef PRECOMPUTED_GRADIENT
layout (set = 0, binding = 6) uniform mediump sampler3D gradient;
#endif
#ifdef ANISOTROPIC_DISTANCE
layout (set = 0, binding = 7) uniform mediump usampler3D distance_map[8];
#else
layout (set = 0, binding = 7) uniform mediump usampler3D distance_map[1];
#endif

layout(location = 0) out vec4 out_color;
layout(depth_greater) out float gl_FragDepth;

vec3 ray_caster_get_back(const in vec3 front, const in vec3 dir) {
  // Use AABB ray-box intersection (simplified due to unit cube [0-1]) to get intersection with back
  vec3 dir_inv = 1.0f / dir;
  vec3 tMin = -front * dir_inv;
  vec3 tMax = (1.0f - front) * dir_inv;
  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);
  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);

  // Return the back intersection
  return tFar * dir + front;
}

float ray_penetration_to_frag_depth(float depth_distance, vec3 ray_entry, float ray_distance) {
  float dist_front = distance(ray_entry, ray_cast_uniform.cam_pos_tex.xyz);
  float relative_depth = (dist_front + ray_distance * depth_distance) / dist_front;
  vec4 pos_view_space_at_depth = vec4(pos_view_space.xyz * relative_depth, 1.0f);
  vec4 p_clip = camera_uniform.proj * pos_view_space_at_depth;
  return p_clip.z / p_clip.w;
}

float get_gradient(vec3 pos, vec3 dim_inv) {
  if (transfer_function_uniform.use_gradient) {
#ifdef PRECOMPUTED_GRADIENT
    // Sample gradient
    float gradient = texture(gradient, pos).x;
#else
    // Gradient on-the-fly using tetrahedron technique http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
    ivec2 k = ivec2(1,-1);
    vec3 gradientDir = (k.xyy * texture(volume, pos + dim_inv * k.xyy).x +
                        k.yyx * texture(volume, pos + dim_inv * k.yyx).x +
                        k.yxy * texture(volume, pos + dim_inv * k.yxy).x +
                        k.xxx * texture(volume, pos + dim_inv * k.xxx).x) * 0.25f;
    float gradient = clamp(length(gradientDir) * transfer_function_uniform.grad_magnitude_modifier, 0, 1);
#endif
    return gradient;
  } else {
    return 1.0f;
  }
}

#ifdef ANISOTROPIC_DISTANCE
int distance_map_idx;
#endif

//int skip(vec3 u, ivec3 u_i, vec3 step_dist_texel_inv) {
//}

//bool sample_and_blend(vec3 pos, float sampling_factor_inv, vec3 dim_inv) {
//
//  return hasAlpha;
//}

void main()
{
  // Initialise
  out_color = vec4(0);

#ifdef DEPTH_ATTACHMENT
  // Read depth
  float frag_depth = subpassLoad(i_depth).x;
  
  // Manual z-test of front face
  vec4 clip_front = camera_uniform.proj * pos_view_space;
  float frag_depth_front = clip_front.z / clip_front.w;
  if (frag_depth < frag_depth_front) {
    discard;
  } else {
    gl_FragDepth = frag_depth;
  }
  
  // Find where ray intersects with depth buffer in texture coordinates
  vec4 clip_at_depth = vec4(clip_front.xyz * frag_depth / frag_depth_front, clip_front.w);
  vec4 pos_at_depth = camera_uniform.view_proj_inv * clip_at_depth;
  pos_at_depth /= pos_at_depth.w;
  vec3 ray_intersect_depth_buffer = (camera_uniform.model_inv * pos_at_depth).xyz + 0.5f;
  float ray_distance_depth_buffer = distance(ray_entry, ray_intersect_depth_buffer);
#else
  gl_FragDepth = 1.0f;
#endif

  // Get ray exit
  vec3 ray_dir = normalize(ray_entry - ray_cast_uniform.cam_pos_tex.xyz);
  vec3 ray_exit = ray_caster_get_back(ray_entry, ray_dir);
  float ray_distance = distance(ray_entry, ray_exit);
#ifdef DEPTH_ATTACHMENT
  if (ray_distance_depth_buffer < ray_distance) {
    // If the depth buffer exit is in front of the normal back intersection, use it instead
    ray_exit = ray_intersect_depth_buffer;
    ray_distance = ray_distance_depth_buffer;
  }
#endif

  // Tests
#ifdef SHOW_RAY_ENTRY
  out_color = vec4(ray_entry, 1.0f); return;
#endif
#ifdef SHOW_RAY_EXIT
  out_color = vec4(ray_exit, 1.0f); return;
#endif

  // Determine number of samples
  ivec3 dim = textureSize(volume, 0);
  int dim_max = max(max(dim.x, dim.y), dim.z);
  int n_steps = int(ceil(float(dim_max) * ray_distance * transfer_function_uniform.sampling_factor));
  vec3 step_volume = ray_dir * ray_distance / (float(n_steps) - 1.0f);
  float sampling_factor_inv = 1.0f / transfer_function_uniform.sampling_factor;

  // This test fixes a performance regression if view is oriented with edge/s of the volume
  // perhaps due to precision issues with the bounding box intersection
  vec3 early_exit_test = ray_entry + step_volume;
  if (any(lessThanEqual(early_exit_test, vec3(0))) || any(greaterThanEqual(early_exit_test, vec3(1)))) {
    return;
  }

#ifndef DISABLE_SKIP
  // Empty space skipping
  ivec3 dim_distance_map = textureSize(distance_map[0], 0);
  vec3 volume_to_distance_map_u = vec3(dim) / (vec3(ray_cast_uniform.block_size));
  ivec3 dim_distance_map_1 = dim_distance_map - 1;
  vec3 step_dist_texel = step_volume * vec3(dim) / vec3(ray_cast_uniform.block_size);
  vec3 step_dist_texel_inv = 1.0f / step_dist_texel;
  int i_min = 0; // furthest sampled step + 1
  ivec3 u_last_alpha = ivec3(0);
#endif

#ifdef SHOW_NUM_SAMPLES
  int num_volume_samples = 0;
  int num_distance_samples = 0;
  int num_empty_samples = 0;
#endif

  // Precompute some constants
  vec3 dim_inv = 1.0f / vec3(dim);
#ifdef ANISOTROPIC_DISTANCE
  distance_map_idx = (ray_dir.z < 0 ? 1 : 0) + (ray_dir.y < 0 ? 2 : 0) + (ray_dir.x < 0 ? 4 : 0);
#endif

  // Step through volume
  bool voxel_occupied = true;
  float ray_penetration = 1.0f; // assume ray goes through
  for (int i = 0; i < n_steps;) {
    vec3 pos = ray_entry + float(i) * step_volume;
    
    #ifndef DISABLE_SKIP
    // Get occupancy/distance map texel coordinate
    vec3 u = volume_to_distance_map_u * pos;
    ivec3 u_i = clamp(ivec3(u), ivec3(0), dim_distance_map_1);

    // Check if space skipping structure should be examined
    if (!voxel_occupied && any(notEqual(u_i, u_last_alpha))) {
      #ifdef SHOW_NUM_SAMPLES
      ++num_distance_samples;
      #endif

      #ifdef ANISOTROPIC_DISTANCE
      uint dist = texelFetch(distance_map[distance_map_idx], u_i, 0).x;
    #else
      uint dist = texelFetch(distance_map[0], u_i, 0).x;
    #endif
      vec3 r = clamp(u_i - u, -1.0, 0.0);
      int i_delta;
      if (dist > 0u) {
    #ifdef BLOCK_SKIP
        // Skip with "block empty space skipping"
        vec3 i_delta_xyz = (step(0.0f, step_dist_texel_inv) + r) * step_dist_texel_inv;
    #else
        // Skip with "chebyshev empty space skipping"
        vec3 i_delta_xyz = (step(0.0f, -step_dist_texel_inv) + sign(step_dist_texel_inv) * float(dist) + r) * step_dist_texel_inv;
    #endif
        i_delta = max(1, int(ceil(min(min(i_delta_xyz.x, i_delta_xyz.y), i_delta_xyz.z))));
        
        // Skip ray forward
        i += i_delta;
      } else {
    #ifdef SHOW_OCCUPANCY
        out_color = vec4(vec3(ray_distance * float(i) / float(n_steps)), 1.0f); return;
    #endif
        // Step backwards
        i_delta = -int(ceil(transfer_function_uniform.sampling_factor));
        // NOTE: The ray is stepped backwards as sample positions just outside of occupied blocks may have some opacity (due to linear sampling of the volume)
        // The artefacts are quite subtle, so this could be optional. For correctness, this is enabled, but obviously causes a slight performance drop.
        // The ray won't ever step back further than the last sampled voxel or make the same back step twice
        
        // Stop skipping and move ray a little bit backwards
        voxel_occupied = true;
        u_last_alpha = u_i;
        i = max(i + i_delta, i_min);
      }
    }
    else
    #endif
    {
      #ifdef SHOW_NUM_SAMPLES
      ++num_volume_samples;
      #endif

      // Map to colour and opacity with a transfer function
      float intensity = texture(volume, pos).x;
      float gradient = get_gradient(pos, dim_inv);
      vec4 color = get_color(intensity, gradient);

      voxel_occupied = color.a > 0.0f;
      if (voxel_occupied) {
        #ifndef DISABLE_SKIP
        u_last_alpha = u_i;
        #endif

        // Correct opacity given sampling factor and multiply colour by alpha
        color.a = clamp(transfer_function_uniform.voxel_alpha_factor * (1.0f - pow(1.0f - color.a, sampling_factor_inv)), 0.0f, 1.0f);  // opacity correction formula
        color.xyz *= color.a;

        // Blend
        out_color = out_color + (1.0f - out_color.a) * color;

        if (out_color.a > 0.99f) {
          // Set ray penetration
          ray_penetration = float(i) / (float(n_steps) - 1.0f);

          #ifndef DISABLE_EARLY_RAY_TERMINATION
          // Early ray termination
          out_color.a = 1.0f;
          break;
          #endif
        }
      } else {
        #ifdef SHOW_NUM_SAMPLES
        ++num_empty_samples;
        #endif
      }
      
      ++i; // move the ray forward
      #ifndef DISABLE_SKIP
      i_min = i;
      #endif
    }

  }



  // Write the depth
  if (out_color.a > 0.0f && ray_penetration < 1.0f) {
    gl_FragDepth = ray_penetration_to_frag_depth(ray_penetration, ray_entry, ray_distance);
  }

#ifdef SHOW_NUM_SAMPLES
  uint n_steps_max = uint(ceil(vec3(dim_max) * sqrt(3.0f)) * transfer_function_uniform.sampling_factor);
//  out_color = vec4(
//    float(num_volume_samples) / float(n_steps_max),
//    float(num_distance_samples) / float(n_steps_max),
//    float(num_empty_samples) / float(n_steps_max),
//    1.0f
//  );
  out_color = vec4(
    vec3(float(num_volume_samples + num_distance_samples) / float(n_steps_max)),
    1.0f
  );
#endif
}
