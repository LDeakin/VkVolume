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

layout(set = TRANSFER_FUNCTION_SET, binding = TRANSFER_FUNCTION_BINDING_UNIFORM) uniform TransferFunctionUniform {
  float sampling_factor;
  float voxel_alpha_factor;
  float grad_magnitude_modifier;
  bool use_gradient;
#ifndef TRANSFER_FUNCTION_BINDING_TEXTURE
  float intensity_min;
  float intensity_range_inv;
  float gradient_min;
  float gradient_range_inv;
#endif
} transfer_function_uniform;

#ifdef TRANSFER_FUNCTION_BINDING_TEXTURE
layout (set = TRANSFER_FUNCTION_SET, binding = TRANSFER_FUNCTION_BINDING_TEXTURE) uniform sampler2D transfer_function;  // rgba
#endif

vec4 get_color(float intensity, float gradient) {
#ifdef TRANSFER_FUNCTION_BINDING_TEXTURE
  // Map intensity and gradient to colour with transfer function
  vec4 color = texture(transfer_function, vec2(intensity, gradient));
#else
  // Map intensity and gradient to colour with simple 2D grayscale equation
  float alphaIntensity = clamp((intensity - transfer_function_uniform.intensity_min) * transfer_function_uniform.intensity_range_inv, 0, 1);
  float alphaGradient = clamp((gradient - transfer_function_uniform.gradient_min) * transfer_function_uniform.gradient_range_inv, 0, 1);
  vec4 color = vec4(alphaIntensity * alphaGradient);
#endif
  return color;
}