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

cmake_minimum_required(VERSION 3.10)

project(vulkan_samples_volume_renderer)

set(VULKAN_SAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Vulkan-samples)
string(LENGTH "${CMAKE_SOURCE_DIR}/" ROOT_PATH_SIZE)
add_definitions(-DROOT_PATH_SIZE=${ROOT_PATH_SIZE})

# Add vulkan framework
set(VKB_VALIDATION_LAYERS OFF CACHE BOOL "Enable validation layers.")
macro(vulkan_samples_pch SRCS PCHCPP)
  if(MSVC AND CMAKE_GENERATOR MATCHES "^Visual Studio")
    set(PCH_NAME "$(IntDir)\\pch.pch")
    # make source files use/depend on PCH_NAME
    set_source_files_properties(${${SRCS}} PROPERTIES COMPILE_FLAGS "/Yupch.h /FIpch.h /Fp${PCH_NAME} /Zm300" OBJECT_DEPENDS "${PCH_NAME}")
    # make PCHCPP file compile and generate PCH_NAME
    set_source_files_properties(${PCHCPP} PROPERTIES COMPILE_FLAGS "/Ycpch.h /Fp${PCH_NAME} /Zm300" OBJECT_OUTPUTS "${PCH_NAME}")
    list(APPEND ${SRCS} "${PCHCPP}")
  endif()
endmacro(vulkan_samples_pch)
add_subdirectory(${VULKAN_SAMPLES}/third_party)
add_subdirectory(${VULKAN_SAMPLES}/framework)

# Add source
add_subdirectory(src)

# Install
install(DIRECTORY "shaders" DESTINATION "./")
install(FILES "README.md" "LICENSE" DESTINATION "./")