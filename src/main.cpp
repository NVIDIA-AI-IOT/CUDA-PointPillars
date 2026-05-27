/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>

#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <array>
#include <algorithm>

#include "pointpillar.hpp"
#include "common/check.hpp"

void GetDeviceInfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string file = ent->d_name;
            if(hasEnding(file, suffix)){
                files.push_back(file.substr(0, file.length()-4));
            }
        }
        closedir(dir);
    } else {
        printf("No such folder: %s.", path);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}

static inline void boxCorners(const pointpillar::lidar::BoundingBox &box, std::array<nvtype::Float3, 8> &corners)
{
    const float cx = box.x, cy = box.y, cz = box.z;
    const float w = box.w, l = box.l, h = box.h;
    const float yaw = box.rt;
    const float hw = w * 0.5f, hl = l * 0.5f, hh = h * 0.5f;
    const float cosr = cosf(yaw), sinr = sinf(yaw);
    auto rot = [&](float x, float y) {
        return nvtype::Float2(x * cosr - y * sinr, x * sinr + y * cosr);
    };

    nvtype::Float2 p0 = rot(-hw, -hl);
    nvtype::Float2 p1 = rot( hw, -hl);
    nvtype::Float2 p2 = rot( hw,  hl);
    nvtype::Float2 p3 = rot(-hw,  hl);

    float z0 = cz - hh;
    float z1 = cz + hh;

    corners[0] = nvtype::Float3(cx + p0.x, cy + p0.y, z0);
    corners[1] = nvtype::Float3(cx + p1.x, cy + p1.y, z0);
    corners[2] = nvtype::Float3(cx + p2.x, cy + p2.y, z0);
    corners[3] = nvtype::Float3(cx + p3.x, cy + p3.y, z0);
    corners[4] = nvtype::Float3(cx + p0.x, cy + p0.y, z1);
    corners[5] = nvtype::Float3(cx + p1.x, cy + p1.y, z1);
    corners[6] = nvtype::Float3(cx + p2.x, cy + p2.y, z1);
    corners[7] = nvtype::Float3(cx + p3.x, cy + p3.y, z1);
}

static inline void interpolateEdge(const nvtype::Float3 &a, const nvtype::Float3 &b, float step, std::vector<nvtype::Float4> &out)
{
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float dz = b.z - a.z;
    const float len = sqrtf(dx * dx + dy * dy + dz * dz);
    if (len <= 1e-6f) {
        out.emplace_back(nvtype::Float4(a.x, a.y, a.z, 255.0f));
        return;
    }
    int num = std::max(2, (int)(len / step) + 1);
    for (int i = 0; i < num; ++i) {
        float t = (float)i / (float)(num - 1);
        out.emplace_back(nvtype::Float4(a.x + t * dx, a.y + t * dy, a.z + t * dz, 255.0f));
    }
}

void SaveBoxesAsPCD(const std::vector<pointpillar::lidar::BoundingBox> &boxes,
                    const float *points_xyzi,
                    int num_points,
                    const std::string &file_name,
                    float edge_step = 0.05f)
{
    std::vector<nvtype::Float4> line_points;
    line_points.reserve((size_t)boxes.size() * 12 * 20);

    for (const auto &b : boxes) {
        // if(b.score < 0.12) 
        //     continue;
        std::array<nvtype::Float3, 8> cs;
        boxCorners(b, cs);
        const int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        for (int e = 0; e < 12; ++e) {
            interpolateEdge(cs[edges[e][0]], cs[edges[e][1]], edge_step, line_points);
        }
    }

    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (!ofs.is_open()) {
        std::cerr << "Output PCD file cannot be opened!" << std::endl;
        return;
    }

    const size_t total_points = (size_t)num_points + line_points.size();
    ofs << "# .PCD v.7 - Point Cloud Data file format\n";
    ofs << "VERSION .7\n";
    ofs << "FIELDS x y z intensity\n";
    ofs << "SIZE 4 4 4 4\n";
    ofs << "TYPE F F F F\n";
    ofs << "COUNT 1 1 1 1\n";
    ofs << "WIDTH " << total_points << "\n";
    ofs << "HEIGHT 1\n";
    ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << total_points << "\n";
    ofs << "DATA ascii\n";

    for (int i = 0; i < num_points; ++i) {
        const float *p = points_xyzi + i * 4;
        ofs << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << "\n";
    }
    for (const auto &q : line_points) {
        ofs << q.x << ", " << q.y << ", " << q.z << ", " << q.w << "\n";
    }
    ofs.close();
    std::cout << "Saved PCD with boxes in: " << file_name << std::endl;
}

void SaveBoxPred(std::vector<pointpillar::lidar::BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

void SaveBoxPredToLabels(std::vector<pointpillar::lidar::BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
        if(box.l>70)
            continue;
          ofs <<"Car 0 0 0 0 0 0 0 ";
          ofs << box.h << " ";
          ofs << box.l << " ";
          ofs << box.w << " ";
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << ((box.rt*180/3.1415926)-180) /180*3.1415926 << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

std::shared_ptr<pointpillar::lidar::Core> create_core() {
    pointpillar::lidar::VoxelizationParameter vp;
    vp.min_range = nvtype::Float3(0.0, -39.68f, -3.0);
    vp.max_range = nvtype::Float3(69.12f, 39.68f, 1.0);
    vp.voxel_size = nvtype::Float3(0.16f, 0.16f, 4.0f);
    vp.grid_size =
        vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 40000;
    vp.max_points_per_voxel = 32;
    vp.max_points = 300000;
    vp.num_feature = 4;

    pointpillar::lidar::PostProcessParameter pp;
    pp.min_range = vp.min_range;
    pp.max_range = vp.max_range;
    pp.feature_size = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = "../model/pointpillar.plan";
    param.lidar_post = pp;
    return pointpillar::lidar::create_core(param);
}

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}

static void help()
{
    printf(
        "Usage: \n"
        "    ./pointpillar in/ out/ --timer\n"
        "    Run pointpillar inference with .bin under in, save .text under out\n"
        "    Optional: --timer, enable timer log\n"
    );
    exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {

    if (argc < 3 || argc > 4)
        help();

    const char *in_dir  = argv[1];
    const char *out_dir  = argv[2];

    const char *value = nullptr;
    bool timer = false;

    if (argc == 4) {
        if (startswith(argv[3], "--timer", &value)) {
            timer = true;
        }
    }

    GetDeviceInfo();

    std::vector<std::string> files;
    getFolderFile(in_dir, files);
    std::cout << "Total " << files.size() << std::endl;

    auto core = create_core();
    if (core == nullptr) {
        printf("Core has been failed.\n");
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
  
    core->print();
    core->set_timer(timer);

    for (const auto & file : files)
    {
        std::string dataFile = std::string(in_dir) + file + ".bin";

        std::cout << "\n<<<<<<<<<<<" <<std::endl;
        std::cout << "Load file: "<< dataFile <<std::endl;

        //load points cloud
        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
        loadData(dataFile.data(), &data, &length);
        buffer.reset((char *)data);
        int points_size = length/sizeof(float)/4;
        std::cout << "Lidar points count: "<< points_size <<std::endl;
    
        auto bboxes = core->forward((float *)buffer.get(), points_size, stream);
        // 过滤掉置信度过低的框
        bboxes.erase(
            std::remove_if(
                bboxes.begin(),
                bboxes.end(),
                [](const pointpillar::lidar::BoundingBox& box) { return box.score < 0.25f; }),
            bboxes.end());

        // 把ROI 区域添加进框里面
        {
            const float min_x = 0.0f,     min_y = -39.68f, min_z = -3.0f;
            const float max_x = 69.12f,   max_y = 39.68f,  max_z =  1.0f;
            const float cx = (min_x + max_x) * 0.5f;
            const float cy = (min_y + max_y) * 0.5f;
            const float cz = min_z;
            const float w  = (max_x - min_x);
            const float l  = (max_y - min_y);
            const float h  = (max_z - min_z);
            pointpillar::lidar::BoundingBox range_box(cx, cy, cz, w, l, h, 0.0f, -1, 1.0f);
            bboxes.push_back(range_box);
        }
        std::cout<<"Detections after NMS: "<< bboxes.size()<<std::endl;

        std::string save_file_name = std::string(out_dir) + file + ".txt";
        SaveBoxPred(bboxes, save_file_name);

        std::string save_file_name_labels = std::string(out_dir) +std::string("labels/")+ file + ".txt";
        SaveBoxPredToLabels(bboxes, save_file_name_labels);

        std::string save_pcd_name = std::string(out_dir) + file + "_boxes.pcd";
        SaveBoxesAsPCD(bboxes, (float *)buffer.get(), points_size, save_pcd_name, 0.05f);

        std::cout << ">>>>>>>>>>>" << std::endl;
    }

    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}
