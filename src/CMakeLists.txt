# Copyright (c) 2019, Lachlan Deakin
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 the "License";
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

find_package(Boost REQUIRED)

set(SOURCES
  compute_distance_map.cpp
  compute_gradient_map.cpp
  compute_occupied_voxel_count.cpp
  load_volume.cpp
  volume_component.cpp
  volume_render_subpass.cpp
  volume_render.cpp
  main.cpp
)

if(MSVC)
  add_executable(vrender WIN32 ${SOURCES})
  set_property(TARGET vrender PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  # set_target_properties(vrender PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE)
else()
  add_executable(vrender ${SOURCES})
endif()

target_include_directories(vrender PRIVATE ${VULKAN_SAMPLES}/app)
target_link_libraries(vrender framework plugins apps Boost::boost)

install(TARGETS vrender DESTINATION "./")