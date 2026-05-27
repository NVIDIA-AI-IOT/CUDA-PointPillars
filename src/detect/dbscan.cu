/*
 * CUDA-accelerated DBSCAN implementation
 * 使用 CUDA 并行加速 DBSCAN 聚类算法
 */

#include "dbscan.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace detect {

// CUDA kernel: 并行计算每个点的邻居数量
__global__ void compute_neighbor_counts_kernel(
    const float* points,      // [n, 3] - 点云数据
    int n,                   // 点的数量
    float eps_xy_squared,    // eps_xy^2 (xy平面距离的平方)
    float eps_z,             // eps_z (z方向距离阈值)
    int* neighbor_counts)    // 输出：每个点的邻居数量
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* query_point = points + idx * 3;
    int count = 0;
    
    for (int i = 0; i < n; ++i) {
        if (i == idx) continue;  // 排除自己
        
        const float* other_point = points + i * 3;
        float dx = query_point[0] - other_point[0];
        float dy = query_point[1] - other_point[1];
        float dz = query_point[2] - other_point[2];
        
        // 分别判断 xy 平面距离和 z 方向距离
        float xy_dist_sq = dx * dx + dy * dy;
        float z_dist_abs = fabsf(dz);
        
        if (xy_dist_sq <= eps_xy_squared && z_dist_abs <= eps_z) {
            count++;
        }
    }
    
    neighbor_counts[idx] = count;
}

// CUDA kernel: 标记核心点和噪声点
__global__ void mark_core_points_kernel(
    const int* neighbor_counts,
    int n,
    int min_points,
    int* labels)  // -1=未访问, 0=噪声, >0=簇ID
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (neighbor_counts[idx] >= min_points) {
        labels[idx] = -2;  // -2 表示核心点（未分配簇）
    } else {
        labels[idx] = 0;   // 噪声点
    }
}

// CUDA kernel: 并行查找邻居索引（使用共享内存优化）
__global__ void find_neighbors_kernel(
    const float* points,
    int n,
    float eps_xy_squared,    // eps_xy^2 (xy平面距离的平方)
    float eps_z,             // eps_z (z方向距离阈值)
    int* neighbor_offsets,    // 每个点的邻居在 neighbor_indices 中的起始位置
    int* neighbor_indices)    // 所有邻居的索引（压缩存储）
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* query_point = points + idx * 3;
    int offset = neighbor_offsets[idx];
    int count = 0;
    
    for (int i = 0; i < n; ++i) {
        if (i == idx) continue;
        
        const float* other_point = points + i * 3;
        float dx = query_point[0] - other_point[0];
        float dy = query_point[1] - other_point[1];
        float dz = query_point[2] - other_point[2];
        
        // 分别判断 xy 平面距离和 z 方向距离
        float xy_dist_sq = dx * dx + dy * dy;
        float z_dist_abs = fabsf(dz);
        
        if (xy_dist_sq <= eps_xy_squared && z_dist_abs <= eps_z) {
            neighbor_indices[offset + count] = i;
            count++;
        }
    }
}

// CUDA 版本的 DBSCAN（使用 GPU 加速邻居搜索，簇扩展在 CPU 上进行）
void dbscan_filter_cuda(
    std::vector<nvtype::Float3>& points_in_box,
    std::vector<std::array<float, 4>>& box_points,
    float eps_xy,
    float eps_z,
    float max_cluster_ratio,
    float z_threshold,
    void* stream_ptr)
{
    if (points_in_box.empty() || points_in_box.size() != box_points.size()) {
        points_in_box.clear();
        box_points.clear();
        return;
    }
    
    const int n = points_in_box.size();
    const int min_points = 2;
    
    if (n == 0) return;
    
    cudaStream_t stream = stream_ptr ? static_cast<cudaStream_t>(stream_ptr) : 0;
    
    // 将点云数据复制到 GPU
    thrust::device_vector<float> d_points(n * 3);
    float* h_points = new float[n * 3];
    for (int i = 0; i < n; ++i) {
        h_points[i * 3 + 0] = points_in_box[i].x;
        h_points[i * 3 + 1] = points_in_box[i].y;
        h_points[i * 3 + 2] = points_in_box[i].z;
    }
    checkRuntime(cudaMemcpyAsync(thrust::raw_pointer_cast(d_points.data()), h_points, 
                    n * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));
    
    // 分配 GPU 内存
    thrust::device_vector<int> d_neighbor_counts(n);
    thrust::device_vector<int> d_labels(n, -1);
    
    float eps_xy_squared = eps_xy * eps_xy;
    
    // Kernel 1: 计算每个点的邻居数量
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    compute_neighbor_counts_kernel<<<blocks, threads_per_block, 0, stream>>>(
        thrust::raw_pointer_cast(d_points.data()),
        n,
        eps_xy_squared,
        eps_z,
        thrust::raw_pointer_cast(d_neighbor_counts.data())
    );
    checkRuntime(cudaGetLastError());
    
    // Kernel 2: 标记核心点和噪声点
    mark_core_points_kernel<<<blocks, threads_per_block, 0, stream>>>(
        thrust::raw_pointer_cast(d_neighbor_counts.data()),
        n,
        min_points,
        thrust::raw_pointer_cast(d_labels.data())
    );
    checkRuntime(cudaGetLastError());
    
    // 同步，确保 kernels 完成
    cudaStreamSynchronize(stream);
    
    // 计算邻居索引的偏移量（使用 thrust::exclusive_scan）
    thrust::device_vector<int> d_neighbor_offsets(n);
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                          d_neighbor_counts.begin(),
                          d_neighbor_counts.end(),
                          d_neighbor_offsets.begin(),
                          0);
    
    // 计算总邻居数
    int total_neighbors = thrust::reduce(thrust::cuda::par.on(stream),
                                        d_neighbor_counts.begin(),
                                        d_neighbor_counts.end());
    
    // 分配邻居索引数组
    thrust::device_vector<int> d_neighbor_indices(total_neighbors);
    
    // Kernel 3: 查找所有邻居索引
    find_neighbors_kernel<<<blocks, threads_per_block, 0, stream>>>(
        thrust::raw_pointer_cast(d_points.data()),
        n,
        eps_xy_squared,
        eps_z,
        thrust::raw_pointer_cast(d_neighbor_offsets.data()),
        thrust::raw_pointer_cast(d_neighbor_indices.data())
    );
    checkRuntime(cudaGetLastError());
    
    // 将结果复制回 CPU
    std::vector<int> h_neighbor_counts(n);
    std::vector<int> h_neighbor_offsets(n);
    std::vector<int> h_neighbor_indices(total_neighbors);
    std::vector<int> h_labels(n);
    
    checkRuntime(cudaMemcpyAsync(h_neighbor_counts.data(), 
                    thrust::raw_pointer_cast(d_neighbor_counts.data()),
                    n * sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaMemcpyAsync(h_neighbor_offsets.data(),
                    thrust::raw_pointer_cast(d_neighbor_offsets.data()),
                    n * sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaMemcpyAsync(h_neighbor_indices.data(),
                    thrust::raw_pointer_cast(d_neighbor_indices.data()),
                    total_neighbors * sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaMemcpyAsync(h_labels.data(),
                    thrust::raw_pointer_cast(d_labels.data()),
                    n * sizeof(int), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    
    // 在 CPU 上进行簇扩展（这部分有复杂的依赖关系，在 CPU 上更简单）
    int cluster_id = 0;
    std::vector<int> labels(n, -1);
    
    // 创建邻居访问函数
    auto get_neighbors = [&](int point_idx) -> std::vector<int> {
        std::vector<int> neighbors;
        int offset = h_neighbor_offsets[point_idx];
        int count = h_neighbor_counts[point_idx];
        neighbors.reserve(count);
        for (int i = 0; i < count; ++i) {
            neighbors.push_back(h_neighbor_indices[offset + i]);
        }
        return neighbors;
    };
    
    // DBSCAN 簇扩展（与原算法相同）
    std::vector<int> seed_set;
    seed_set.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        if (h_labels[i] != -2) continue;  // 只处理核心点
        if (labels[i] != -1) continue;    // 已访问过
        
        std::vector<int> neighbors = get_neighbors(i);
        if (neighbors.size() < min_points) {
            labels[i] = 0;
            continue;
        }
        
        cluster_id++;
        labels[i] = cluster_id;
        
        seed_set.clear();
        seed_set.insert(seed_set.end(), neighbors.begin(), neighbors.end());
        
        for (size_t j = 0; j < seed_set.size(); ++j) {
            int neighbor_idx = seed_set[j];
            
            if (labels[neighbor_idx] == cluster_id) {
                continue;
            }
            
            if (labels[neighbor_idx] == 0) {
                labels[neighbor_idx] = cluster_id;
            } else if (labels[neighbor_idx] != -1) {
                continue;
            }
            
            labels[neighbor_idx] = cluster_id;
            
            std::vector<int> neighbor_neighbors = get_neighbors(neighbor_idx);
            if (neighbor_neighbors.size() >= min_points) {
                for (int nn : neighbor_neighbors) {
                    if (labels[nn] == -1 || labels[nn] == 0) {
                        seed_set.push_back(nn);
                    }
                }
            }
        }
    }
    
    // 统计和过滤（与原算法相同）
    std::vector<int> cluster_counts(cluster_id + 1, 0);
    std::vector<float> cluster_max_z(cluster_id + 1, -std::numeric_limits<float>::infinity());
    
    for (int i = 0; i < n; ++i) {
        if (labels[i] > 0) {
            cluster_counts[labels[i]]++;
            cluster_max_z[labels[i]] = std::max(cluster_max_z[labels[i]], points_in_box[i].z);
        }
    }
    
    int max_cluster_size = 0;
    int max_cluster_id = 0;
    for (int cid = 1; cid <= cluster_id; ++cid) {
        if (cluster_counts[cid] > max_cluster_size) {
            max_cluster_size = cluster_counts[cid];
            max_cluster_id = cid;
        }
    }
    
    float main_cluster_max_z = (max_cluster_id > 0) ? cluster_max_z[max_cluster_id] : 0.0f;
    float z_limit = main_cluster_max_z + z_threshold;
    float size_threshold = max_cluster_size * max_cluster_ratio;
    
    std::vector<nvtype::Float3> filtered_points;
    std::vector<int> filtered_indices;
    filtered_points.reserve(n);
    filtered_indices.reserve(n);
    
    for (int i = 0; i < n; ++i) {
        if (labels[i] > 0) {
            int cid = labels[i];
            int cluster_size = cluster_counts[cid];
            
            if (cluster_size >= size_threshold || points_in_box[i].z <= z_limit) {
                filtered_points.push_back(points_in_box[i]);
                filtered_indices.push_back(i);
            }
            // else //  调试用
            // {
            //     // 不满足条件但仍需保留，将强度改为 -2 标记
            //     box_points[i][3] = -2.0f;
            //     filtered_points.push_back(points_in_box[i]);
            //     filtered_indices.push_back(i);
            // }
        }
    }
    
    if (filtered_points.empty()) {
        points_in_box.clear();
        box_points.clear();
        delete[] h_points;
        return;
    }
    
    points_in_box = std::move(filtered_points);
    std::vector<std::array<float, 4>> filtered_box_points;
    filtered_box_points.reserve(filtered_indices.size());
    for (int idx : filtered_indices) {
        filtered_box_points.push_back(box_points[idx]);
    }
    box_points = std::move(filtered_box_points);
    
    delete[] h_points;
}

} // namespace detect

