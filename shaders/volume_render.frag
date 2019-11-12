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

precision highp float;

#ifdef DEPTH_ATTACHMENT
layout (input_attachment_index = 0, binding = 0) uniform subpassInput i_depth;
#endif

layout(location = 0) in vec4 pos_view_space;
layout(location = 1) in vec3 ray_entry;

layout(set = 0, binding = 1) uniform VolumeRenderUniform {
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
    float grad_magnitude_modifier;
#ifndef TRANSFER_FUNCTION_TEXTURE
    float intensity_min;
    float intensity_max;
    float gradient_min;
    float gradient_max;
#endif
} volume_uniform;

layout (set = 0, binding = 2) uniform mediump sampler3D volume;
#ifdef PRECOMPUTED_GRADIENT
layout (set = 0, binding = 3) uniform mediump sampler3D gradient;
#endif
#ifdef TRANSFER_FUNCTION_TEXTURE
layout (set = 0, binding = 4) uniform mediump sampler2D transfer_function;
#endif
layout (set = 0, binding = 5) uniform mediump usampler3D distance_map;

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
  float dist_front = distance(ray_entry, volume_uniform.cam_pos_tex.xyz);
  float relative_depth = (dist_front + ray_distance * depth_distance) / dist_front;
  vec4 pos_view_space_at_depth = vec4(pos_view_space.xyz * relative_depth, 1.0f);
  vec4 p_clip = volume_uniform.proj * pos_view_space_at_depth;
  return p_clip.z / p_clip.w;
}

void main()
{
  // Initialise
  out_color = vec4(0);

#ifdef DEPTH_ATTACHMENT
  // Read depth
  float frag_depth = subpassLoad(i_depth).x;
  
  // Manual z-test of front face
  vec4 clip_front = volume_uniform.proj * pos_view_space;
  float frag_depth_front = clip_front.z / clip_front.w;
  if (frag_depth < frag_depth_front) {
    discard;
  } else {
    gl_FragDepth = frag_depth;
  }
  
  // Find where ray intersects with depth buffer in texture coordinates
  vec4 clip_at_depth = vec4(clip_front.xyz * frag_depth / frag_depth_front, clip_front.w);
  vec4 pos_at_depth = volume_uniform.view_proj_inv * clip_at_depth;
  pos_at_depth /= pos_at_depth.w;
  vec3 ray_intersect_depth_buffer = (volume_uniform.model_inv * pos_at_depth).xyz + 0.5f;
  float ray_distance_depth_buffer = distance(ray_entry, ray_intersect_depth_buffer);
#else
  gl_FragDepth = 1.0f;
#endif

  // Get ray exit
  vec3 ray_dir = normalize(ray_entry - volume_uniform.cam_pos_tex.xyz);
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
  int n_steps = int(ceil(float(dim_max) * ray_distance * volume_uniform.sampling_factor));
  vec3 step_volume = ray_dir * ray_distance / (float(n_steps) - 1.0f);
  float sampling_factor_inv = 1.0f / volume_uniform.sampling_factor;

  // This test fixes a performance regression if view is oriented with edge/s of the volume
  // perhaps due to precision issues with the bounding box intersection
  vec3 early_exit_test = ray_entry + step_volume;
  if (any(lessThanEqual(early_exit_test, vec3(0))) || any(greaterThanEqual(early_exit_test, vec3(1)))) {
    return;
  }

#ifndef DISABLE_SKIP
  // Empty space skipping
  ivec3 dim_distance_map = textureSize(distance_map, 0);
  ivec3 block_size = dim / dim_distance_map;  // NOTE: scalar in paper, but can be a vector
  vec3 volume_to_distance_map = vec3(dim) / (vec3(block_size) * vec3(dim_distance_map));
  vec3 step_dist_texel = step_volume * vec3(dim) / vec3(block_size);
  vec3 step_dist_texel_inv = 1.0f / step_dist_texel;
  bool skipping = true;
  int i_last_alpha = 0; // current furthest "occupied" step
  int i_resume = int(ceil(1.5f * volume_uniform.sampling_factor * max(max(block_size.x, block_size.y), block_size.z)));
#endif

#ifdef SHOW_NUM_SAMPLES
  int num_volume_samples = 0;
  int num_distance_samples = 0;
#endif

  // Precompute some constants
  vec3 dim_inv = 1.0f / vec3(dim);
#ifndef TRANSFER_FUNCTION_TEXTURE
  float intensity_range_inv = 1.0f / (volume_uniform.intensity_max - volume_uniform.intensity_min);
  float gradient_range_inv = 1.0f / (volume_uniform.gradient_max - volume_uniform.gradient_min);
#endif

  // Step through volume
  float ray_penetration = 1.0f; // assume ray goes through
  for (int i = 0; i < n_steps;) {
    vec3 pos = ray_entry + float(i) * step_volume;

#ifndef DISABLE_SKIP
    if (skipping) {
      vec3 pos_distance_map = volume_to_distance_map * pos;
      vec3 u = pos_distance_map * vec3(dim_distance_map);
      ivec3 u_i = ivec3(clamp(floor(u), ivec3(0), dim_distance_map - 1));
      uint dist = texelFetch(distance_map, u_i, 0).x;
      vec3 r = -fract(u);
      // uint dist = texture(distance_map, pos_distance_map).x;
#ifdef SHOW_NUM_SAMPLES
      ++num_distance_samples;
#endif
      if (dist > 0u) {
#ifdef BLOCK_SKIP
        // Skip with "block empty space skipping"
        vec3 n_iter = (step(0.0f, step_dist_texel_inv) + r) * step_dist_texel_inv;
#else
        // Skip with "chebyshev empty space skipping"
        vec3 n_iter = (step(0.0f, -step_dist_texel_inv) + sign(step_dist_texel_inv) * float(dist) + r) * step_dist_texel_inv;
#endif
        int skips = max(1, int(ceil(min(min(n_iter.x, n_iter.y), n_iter.z))));
        i += skips;
      } else {
#ifdef SHOW_OCCUPANCY
        out_color = vec4(vec3(ray_distance * float(i) / float(n_steps)), 1.0f); return;
#endif
        // Stop skipping and step back
        skipping = false;
        int i_backwards = max(i - int(ceil(volume_uniform.sampling_factor)), i_last_alpha);
        i_last_alpha = i + 1;
        i = i_backwards;
        // NOTE: The ray is stepped backwards as sample positions just outside of occupied blocks may have some opacity (due to linear sampling of the volume)
        // The artefacts are quite subtle, so this could be optional. For correctness, this is enabled, but obviously causes a slight performance drop.
        // The ray won't ever step back further than the last sampled voxel or make the same back step twice, say if i_resume is abnormally tiny relative to the region size
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
      float gradient = clamp(length(gradientDir) * volume_uniform.grad_magnitude_modifier, 0, 1);
#endif

#ifdef TRANSFER_FUNCTION_TEXTURE
      // Map intensity and gradient to colour with transfer function
      vec4 color = texture(transfer_function, vec2(intensity, gradient));
#else
      // Map intensity and gradient to colour with simple 2D grayscale equation
      float alphaIntensity = clamp((intensity - volume_uniform.intensity_min) * intensity_range_inv, 0, 1);
      float alphaGradient = clamp((gradient - volume_uniform.gradient_min) * gradient_range_inv, 0, 1);
      vec4 color = vec4(alphaIntensity * alphaGradient);
#endif

      if (color.a > 0.0f) {
          // Correct opacity given sampling factor and multiply colour by alpha
          color.a = clamp(volume_uniform.voxel_alpha_factor * (1.0f - pow(1.0f - color.a, sampling_factor_inv)), 0.0f, 1.0f);  // opacity correction formula
          color.xyz *= color.a;

          // Blend
          out_color = out_color + (1.0f - out_color.a) * color;
#ifndef DISABLE_SKIP
          i_last_alpha = i;
#endif

          if (out_color.a > 0.99f) {
            // Set ray penetration
            ray_penetration = float(i) / (float(n_steps) - 1.0f);

#ifndef DISABLE_EARLY_RAY_TERMINATION
            // Early ray termination
            out_color.a = 1.0f;
            break;
#endif
          }

      } 
#ifndef DISABLE_SKIP
      else if (i >= (i_last_alpha + i_resume)) {
          // Resume empty space skipping
  #ifdef SHOW_RESUME
          out_color = vec4(1.0f, 0.0f, 0.0f, 1.0f); return;
  #endif
          skipping = true;
          i_last_alpha = i + 1; // skipping won't jump back further than this if next block is occupied
      }
#endif

      // Move the ray forward
      ++i;
    }

  }

  // Write the depth
  if (out_color.a > 0.0f && ray_penetration < 1.0f) {
    gl_FragDepth = ray_penetration_to_frag_depth(ray_penetration, ray_entry, ray_distance);
  }

#ifdef SHOW_NUM_SAMPLES
  uint n_steps_max = uint(ceil(vec3(dim_max) * sqrt(3.0f)) * volume_uniform.sampling_factor);
  out_color = vec4(vec3(float(num_volume_samples + num_distance_samples) / float(n_steps_max)), 1.0f); return;
#endif
}
