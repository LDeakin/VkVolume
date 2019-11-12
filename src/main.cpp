/* Copyright (c) 2019, Arm Limited and Contributors
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

#include "common/logging.h"
#include "platform/platform.h"

#include "volume_render.h"
#if defined(VK_USE_PLATFORM_WIN32_KHR)
#	include "platform/windows/windows_platform.h"
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                     PSTR lpCmdLine, INT nCmdShow)
{
	vkb::WindowsPlatform platform{hInstance, hPrevInstance,
	                              lpCmdLine, nCmdShow};
#elif defined(VK_USE_PLATFORM_DISPLAY_KHR)
#	include "platform/linux/linux_d2d_platform.h"
int main(int argc, char *argv[])
{
	vkb::LinuxD2DPlatform platform{argc, argv};
#else
#	include "platform/linux/linux_platform.h"
int main(int argc, char *argv[])
{
	vkb::LinuxPlatform platform{argc, argv};
#endif

	try
	{
		auto sample = create_volume_render();
		sample->set_name("volume_render");
		sample->parse_options(platform.get_arguments());

		if (platform.initialize(std::move(sample)))
		{
			platform.main_loop();
			platform.terminate(vkb::ExitCode::Success);
		}
		else
		{
			platform.terminate(vkb::ExitCode::UnableToRun);
		}
		platform.terminate(vkb::ExitCode::Success);
	}
	catch (const std::exception &e)
	{
		LOGE(e.what());
		platform.terminate(vkb::ExitCode::FatalError);
	}
}
