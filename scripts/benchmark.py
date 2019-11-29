#!/usr/bin/env python3

import subprocess
import re
import pandas as pd

app = "C:/dev/VkVolume/out/build/x64-Release/src/vrender.exe"
cwd = "C:/dev/VkVolume"

width = 1200
height = 1200
frames = 1000

params = ["fn", "image", "imin", "imax", "gmin", "gmax"]

#images = [
#          ["present_492x492x442.uint16", "present", 0.071, 1.0, 0.0, 0.0]
#         # ,["present_492x492x442.uint16", "present", 0.306, 1.0, 0.0, 0.0]
#         ,["present_492x492x442.uint16", "present", 0.071, 1.0, 0.06, 0.1]
#         ,["stag_beetle_832x832x494.uint16", "beetle", 0.086, 1.0, 0.0, 0.0]
#         ,["stag_beetle_832x832x494.uint16", "beetle", 0.086, 1.0, 0.1, 0.3]
#         # ,["kingsnake_1024x1024x795.uint8", "snake", 0.099, 1.0, 0.0, 0.0]
#         ,["kingsnake_1024x1024x795.uint8", "snake", 0.4, 0.8, 0.0, 0.0]
#         ,["kingsnake_1024x1024x795.uint8", "snake", 0.2, 0.8, 0.06, 0.12]
#         ]

images = [
          ["present_492x492x442.uint16", "present", 0.071, 1.0, 0.0, 0.0]
         ,["present_492x492x442.uint16", "present", 0.071, 1.0, 0.06, 0.1]
         ,["stag_beetle_832x832x494.uint16", "beetle", 0.086, 1.0, 0.0, 0.0]
         ,["stag_beetle_832x832x494.uint16", "beetle", 0.086, 1.0, 0.1, 0.3]
         ,["kingsnake_1024x1024x795.uint8", "snake", 0.4, 0.8, 0.0, 0.0]
         ,["kingsnake_1024x1024x795.uint8", "snake", 0.2, 0.8, 0.06, 0.12]
         ]

images = [{ p: i for p, i in zip(params, image) } for image in images] # Convert to dict

def get_timing(image, b, skipmode):
    output = "FAIL"
    try:
        output = subprocess.Popen(
            [app,
             "--width={}".format(width),
             "--height={}".format(height),
             "--benchmark={}".format(frames),
             "--imin={}".format(image["imin"]),
             "--imax={}".format(image["imax"]),
             "--gmin={}".format(image["gmin"]),
             "--gmax={}".format(image["gmax"]),
             "--blocksize={}".format(b),
             "--skipmode={}".format(skipmode),
             image["fn"]],
            cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode('utf-8')
        m = re.search(r"ran [\d]+ frames, averaged ([\d\.]+) fps", output)
        framerate = m.group(1)
        m = re.search(r"Updated occupancy/distance map in ([\d\.]+)ms", output)
        distance_map_time_ms = m.group(1)
        m = re.search(r"Occupied voxels: ([\d\.]+)%", output)
        occupied_voxel_percent = m.group(1)
        return (framerate, distance_map_time_ms, occupied_voxel_percent)
    except:
        print(output)
        return None

def benchmark_block_sizes(skipmode, bs):
    results = []
    for image in images:
        print(image)
        for b in bs:
            if (skipmode != 0 or b == bs[0]):
                result = get_timing(image, b, skipmode)
            if result:
                (framerate, distance_map_time_ms, occupied_voxel_percent) = result
                image.update({
                        "skipmode": int(skipmode),
                        "blocksize": int(b),
                        "occupancy": float(occupied_voxel_percent),
                        "framerate": float(framerate),
                        "update": float(distance_map_time_ms)
                    })
                print("\t", skipmode, b, framerate, distance_map_time_ms, occupied_voxel_percent)
                results.append(image.copy())
    
    columns = ["image", "skipmode", "blocksize", "occupancy", "framerate", "update", "imin", "imax", "gmin", "gmax"]
    df = pd.DataFrame(results, columns=columns)
    print(df.to_string())
    df.to_csv("benchmark_results_{}.csv".format(skipmode), index=False)

# Block size benchmarking, was just run once
for skipmode in [0, 1, 2, 3]:
  bs = [2, 3, 4, 5, 6]
  benchmark_block_sizes(skipmode, bs)
