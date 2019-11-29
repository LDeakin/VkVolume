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

struct TransferFunctionUniform
{
	float sampling_factor;
	float voxel_alpha_factor;
	float grad_magnitude_modifier;
  VkBool32 use_gradient;
#ifndef TRANSFER_FUNCTION_TEXTURE
  float intensity_min;
  float intensity_range_inv;
  float gradient_min;
  float gradient_range_inv;
#endif
};