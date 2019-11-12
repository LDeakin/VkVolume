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

#include "load_volume.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <boost/endian/conversion.hpp>

#include <glm/gtx/transform.hpp>

#undef min
#undef max

LoadVolume::Header LoadVolume::load_header(std::string filename_header)
{
	std::ifstream file(filename_header);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open header file");
	}

	Header      header;
	std::string line;

	// Example header file
	/*
832 832 494 # extents
0.001 0.001 0.001 # voxel size
400.0 2538.0 # normalisation range
uint16_t little # data type and endianness (big or little)
1 0 0 90 # rotation axis and angle (degrees)
  */

	// FIXME: Error/sanity checking

	// Extent
	std::getline(file, line);
	std::istringstream ss(line);
	ss >> header.extent.width >> header.extent.height >> header.extent.depth;

	// Voxel size
	std::getline(file, line);
	ss = std::istringstream(line);
	ss >> header.voxel_size.x >> header.voxel_size.y >> header.voxel_size.z;

	// Normalisation range
	std::getline(file, line);
	ss = std::istringstream(line);
	ss >> header.normalisation_range.x >> header.normalisation_range.y;

	// Data type and endianness
	std::getline(file, line);
	ss = std::istringstream(line);
	ss >> header.type >> header.endianness;

	// Angle axis image rotation, angle in degrees
	std::getline(file, line);
	ss = std::istringstream(line);
	glm::vec4 angle_axis;
	ss >> angle_axis.x >> angle_axis.y >> angle_axis.z >> angle_axis.w;

	// Compute image transformation
	glm::vec3 physical_size = header.voxel_size * glm::vec3(header.extent.width, header.extent.height, header.extent.depth);
	header.image_transform  = glm::rotate(glm::radians(angle_axis.w), glm::vec3(angle_axis.xyz)) * glm::scale(physical_size);

	return header;
}

std::vector<uint8_t> LoadVolume::load_data(std::string filename_data, const Header &header)
{
	if (header.type == "uint8_t")
	{
		return load_data_impl<uint8_t>(filename_data, header);
	}
	else if (header.type == "int8_t")
	{
		return load_data_impl<int8_t>(filename_data, header);
	}
	else if (header.type == "uint16_t")
	{
		return load_data_impl<uint16_t>(filename_data, header);
	}
	else if (header.type == "int16_t")
	{
		return load_data_impl<int16_t>(filename_data, header);
	}
	else
	{
		throw std::runtime_error("unsupported image data type");
	}
}

template <typename T>
std::vector<uint8_t> LoadVolume::load_data_impl(std::string filename_data, const Header &header)
{
	size_t         n_voxels = static_cast<size_t>(header.extent.width) * static_cast<size_t>(header.extent.height) * static_cast<size_t>(header.extent.depth);
	std::vector<T> image_data(n_voxels);
	size_t         file_size = image_data.size() * sizeof(T);

	// Load volume into memory
	std::ifstream file(filename_data, std::ios::binary);
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open data file");
	}
	// Get file size
	file.seekg(0, std::ios::end);
	size_t file_size_actual = file.tellg();
	if (file_size_actual != file_size)
	{
		throw std::runtime_error("File size does not match expected size for the given image format/dimensions");
	}
	file.seekg(0, std::ios::beg);

	// Read into memory
	size_t byte_pos   = 0;
	size_t bytes_left = file_size;
	while (bytes_left > 0)
	{
		size_t bytes_read = std::min(bytes_left, size_t(1e8));        // 1e8 = 100MB
		file.read(reinterpret_cast<char *>(image_data.data()) + byte_pos, bytes_read);
		if (!file)
		{
			throw std::runtime_error("File error");
		}
		byte_pos += bytes_read;
		bytes_left -= bytes_read;
	}
	file.clear();

	// Change to machine endianness
	bool big_endian = header.endianness == "big";
	for (T &i : image_data)
	{
		if (big_endian)
		{
			boost::endian::big_to_native_inplace(i);
		}
		else
		{
			boost::endian::little_to_native_inplace(i);
		}
	}

	// Convert to uint8_t
	auto                 min = header.normalisation_range.x;
	auto                 max = header.normalisation_range.y;
	std::vector<uint8_t> volume_data(n_voxels);
	std::transform(image_data.begin(), image_data.end(), volume_data.begin(),
	               [min, max](T v) -> uint8_t { return static_cast<uint8_t>(std::numeric_limits<uint8_t>::max() * std::max(0.0f, std::min(1.0f, (static_cast<float>(v) - min) / (max - min)))); });

	return volume_data;
}
