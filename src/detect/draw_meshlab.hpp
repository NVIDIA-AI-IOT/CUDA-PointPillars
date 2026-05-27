/*
 * Draw bounding boxes on point cloud and save as PCD file
 * Similar functionality to Python's render_boxes_to_pcd
 */

#pragma once

#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <cmath>
#include "process.hpp"
#include "common/dtype.hpp"
#include "config.hpp"

namespace detect {

/**
 * Compute the 8 corners of a 3D bounding box
 * @param box The bounding box
 * @param corners Output array of 8 corner points
 */
static inline void boxCorners(const detect::ProcessingBox &box, 
                               std::array<nvtype::Float3, 8> &corners) {
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

/**
 * Interpolate points along an edge of a bounding box
 * @param a Start point
 * @param b End point
 * @param step Step size for interpolation
 * @param out Output vector of interpolated points with intensity
 */
static inline void interpolateEdge(const nvtype::Float3 &a, 
                                    const nvtype::Float3 &b, 
                                    float step, 
                                    std::vector<nvtype::Float4> &out) {
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

inline bool save_pcd_file(const std::string &file_name,
                   const std::vector<std::array<float, 4>> &points) {
    std::ofstream ofs(file_name, std::ios::out);
    if (!ofs.is_open()) {
        printf("Output PCD file cannot be opened: %s\n",  file_name.c_str());
        return false;
    }

    ofs << "# .PCD v.7 - Point Cloud Data file format\n";
    ofs << "VERSION .7\n";
    ofs << "FIELDS x y z intensity\n";
    ofs << "SIZE 4 4 4 4\n";
    ofs << "TYPE F F F F\n";
    ofs << "COUNT 1 1 1 1\n";
    ofs << "WIDTH " << points.size() << "\n";
    ofs << "HEIGHT 1\n";
    ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << points.size() << "\n";
    ofs << "DATA ascii\n";

    for (const auto &p : points) {
        ofs << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << "\n";
    }

    ofs.close();
    return true;
}

/**
 * Draw parallel lines on the bottom face of the last bounding box
 * Lines are drawn along x direction, spaced 5 meters apart in y direction
 */
static inline void drawParallelLinesOnBottomFace_l(
    const std::array<nvtype::Float3, 8> &cs,
    float edge_step,
    std::vector<std::array<float, 4>> &out_points) {
    
    const nvtype::Float3 &bottom_left = cs[0];
    const nvtype::Float3 &bottom_right = cs[1];
    const nvtype::Float3 &top_left = cs[3];
    
    const float dx = bottom_right.x - bottom_left.x;
    const float dy = bottom_right.y - bottom_left.y;
    const float x_length = sqrtf(dx * dx + dy * dy);
    
    const float dy_perp = top_left.y - bottom_left.y;
    const float dx_perp = top_left.x - bottom_left.x;
    const float y_length = sqrtf(dx_perp * dx_perp + dy_perp * dy_perp);
    
    if (y_length > 1e-6f && x_length > 1e-6f) {
        const float norm_dx = dx / x_length;
        const float norm_dy = dy / x_length;
        const float fixed_y = bottom_left.y;
        const float fixed_z = bottom_left.z;
        
        nvtype::Float3 x_point(bottom_left.x, fixed_y, fixed_z);
        nvtype::Float3 x_end(x_point.x + norm_dx * x_length, fixed_y, fixed_z);
        
        std::vector<nvtype::Float4> line_points;
        interpolateEdge(x_point, x_end, edge_step, line_points);
        for (const auto &sp : line_points) {
            out_points.push_back({sp.x, sp.y, sp.z, -4.0f});
        }
        
        const float segment_interval = 1.0f;
        int num_segments = static_cast<int>(y_length / segment_interval);
        
        for (int seg = 1; seg <= num_segments; ++seg) {
            float y_offset = static_cast<float>(seg) * segment_interval;
            if (y_offset > y_length) y_offset = y_length;
            
            nvtype::Float3 line_start(x_point.x, x_point.y + y_offset, x_point.z);
            nvtype::Float3 line_end(x_end.x, x_end.y + y_offset, x_end.z);
            
            std::vector<nvtype::Float4> seg_line_points;
            interpolateEdge(line_start, line_end, edge_step, seg_line_points);
            for (const auto &sp : seg_line_points) {
                out_points.push_back({sp.x, sp.y, sp.z, -4.0f});
            }
        }
    }
}

static inline void drawParallelLinesOnBottomFace_w(
    const std::array<nvtype::Float3, 8> &cs,
    float edge_step,
    std::vector<std::array<float, 4>> &out_points) {
        
    const nvtype::Float3 &bottom_left = cs[0];
    const nvtype::Float3 &bottom_right = cs[1];
    const nvtype::Float3 &top_left = cs[3];
    
    const float dx = bottom_right.x - bottom_left.x;
    const float dy = bottom_right.y - bottom_left.y;
    const float x_length = sqrtf(dx * dx + dy * dy);
    
    const float dy_perp = top_left.y - bottom_left.y;
    const float dx_perp = top_left.x - bottom_left.x;
    const float y_length = sqrtf(dx_perp * dx_perp + dy_perp * dy_perp);
    
    if (y_length > 1e-6f && x_length > 1e-6f) {
        // 使用垂直于底边的方向画线（与 drawParallelLinesOnBottomFace_l 垂直）
        const float norm_dx = dx_perp / y_length;  // 垂直方向单位向量
        const float norm_dy = dy_perp / y_length;
        const float fixed_x = bottom_left.x;
        const float fixed_z = bottom_left.z;
        
        // nvtype::Float3 x_point(bottom_left.x, fixed_y, fixed_z);
        // nvtype::Float3 x_end(x_point.x + norm_dx * x_length, fixed_y, fixed_z);
        nvtype::Float3 x_point(fixed_x, bottom_left.y, fixed_z);
        nvtype::Float3 x_end(fixed_x, bottom_left.y+ norm_dy * y_length, fixed_z);
        
        std::vector<nvtype::Float4> line_points;
        interpolateEdge(x_point, x_end, edge_step, line_points);
        for (const auto &sp : line_points) {
            out_points.push_back({sp.x, sp.y, sp.z, -4.0f});
        }
        
        //const float segment_interval = 1.0f;
        //int num_segments = static_cast<int>(y_length / segment_interval);
        int num_segments = get_config().lane_count;
        const float segment_interval =  x_length / num_segments ;
        
        for (int seg = 1; seg <= num_segments; ++seg) {
            float x_offset = static_cast<float>(seg) * segment_interval;
            if (x_offset > x_length) x_offset = x_length;
            
            nvtype::Float3 line_start(x_point.x + x_offset, x_point.y, x_point.z);
            nvtype::Float3 line_end(x_end.x + x_offset, x_end.y, x_end.z);
            
            std::vector<nvtype::Float4> seg_line_points;
            interpolateEdge(line_start, line_end, edge_step, seg_line_points);
            for (const auto &sp : seg_line_points) {
                out_points.push_back({sp.x, sp.y, sp.z, -4.0f});
            }
        }
    }
}


/**
 * Save point cloud with bounding boxes to a PCD file
 * Similar to Python's render_boxes_to_pcd function
 * @param boxes Vector of bounding boxes to render
 * @param points_xyzi Point cloud data as x,y,z,intensity
 * @param num_points Number of points in the point cloud
 * @param file_name Output file path
 * @param edge_step Step size for edge interpolation (default: 0.05)
 */
inline bool SaveBoxesAsPCD(const std::vector<detect::ProcessingBox> &boxes,
                    const float *points_xyzi,
                    int num_points,
                    const std::string &file_name,
                    float edge_step,
                    std::vector<std::array<float, 4>> &out_points,
                    bool draw_speed_line = false) {
    out_points.clear();
    const size_t approx_line_points = static_cast<size_t>(boxes.size()) * 12 * 20;
    out_points.reserve(static_cast<size_t>(num_points) + approx_line_points);
    for (int i = 0; i < num_points; ++i) {
        const float *p = points_xyzi + i * 4;
        out_points.push_back({p[0], p[1], p[2], p[3]});
    }

    std::vector<nvtype::Float4> edge_points;
    edge_points.reserve(12 * 20);
    for (size_t idx = 0; idx < boxes.size(); ++idx) {
        const auto &b = boxes[idx];
        std::array<nvtype::Float3, 8> cs;
        boxCorners(b, cs);
        const int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        edge_points.clear();
        for (int e = 0; e < 12; ++e) {
            interpolateEdge(cs[edges[e][0]], cs[edges[e][1]], edge_step, edge_points);
        }
        float intensity = -2.0f ;
        if(idx == boxes.size() - 1) 
        {
            intensity = -1.0f ;
            
            // 在最后一个框的底沿画平行线段
            drawParallelLinesOnBottomFace_l(cs, edge_step, out_points);
            drawParallelLinesOnBottomFace_w(cs, edge_step, out_points);
        }
        for (const auto &q : edge_points) {
            out_points.push_back({q.x, q.y, q.z, intensity});
        }
        
    }

    // Add line1 segment with intensity -2 (from config)
    const auto& line1_config = get_config().line1_config;
    nvtype::Float3 line1_start(line1_config.start_x, line1_config.start_y, line1_config.start_z);
    nvtype::Float3 line1_end(line1_config.end_x, line1_config.end_y, line1_config.end_z);
    
    std::vector<nvtype::Float4> line1_points;
    interpolateEdge(line1_start, line1_end, edge_step, line1_points);
    
    // Add line1 points with intensity -2
    for (const auto &p : line1_points) {
        out_points.push_back({p.x, p.y, p.z, -2.0f});
    }
    
    // Add line2 segment with intensity -2 (from config)
    const auto& line2_config = get_config().line2_config;
    nvtype::Float3 line2_start(line2_config.start_x, line2_config.start_y, line2_config.start_z);
    nvtype::Float3 line2_end(line2_config.end_x, line2_config.end_y, line2_config.end_z);
    
    std::vector<nvtype::Float4> line2_points;
    interpolateEdge(line2_start, line2_end, edge_step, line2_points);
    
    // Add line2 points with intensity -2
    for (const auto &p : line2_points) {
        out_points.push_back({p.x, p.y, p.z, -2.0f});
    }
    
    // Add speed_line segment with intensity -2 (only if draw_speed_line is true)
    if (draw_speed_line) {
        const auto& speed_line_config = get_config().speed_line_config;
        nvtype::Float3 speed_line_start(speed_line_config.start_x, speed_line_config.start_y, speed_line_config.start_z);
        nvtype::Float3 speed_line_end(speed_line_config.end_x, speed_line_config.end_y, speed_line_config.end_z);
        
        std::vector<nvtype::Float4> speed_line_points;
        interpolateEdge(speed_line_start, speed_line_end, edge_step, speed_line_points);
        
        // Add speed_line points with intensity -2
        for (const auto &p : speed_line_points) {
            out_points.push_back({p.x, p.y, p.z, -2.0f});
        }
    }
    
    if (!file_name.empty()) {
        return save_pcd_file(file_name, out_points);
    }
    return true;
}

/**
 * Append bounding boxes to existing points array (more efficient, no format conversion)
 * Directly appends box edges to the points vector without clearing it
 * @param boxes Vector of bounding boxes to render
 * @param points Existing point cloud data (will be appended to)
 * @param edge_step Step size for edge interpolation
 */
inline void SaveBoxes(const std::vector<detect::ProcessingBox> &boxes,
                      std::vector<std::array<float, 4>> &points,
                      float edge_step) {
    if (boxes.empty()) {
        return;
    }
    
    // Reserve space for edge points (approximate)
    const size_t approx_edge_points = static_cast<size_t>(boxes.size()) * 12 * 20;
    points.reserve(points.size() + approx_edge_points);
    
    std::vector<nvtype::Float4> edge_points;
    edge_points.reserve(12 * 20);
    
    for (size_t idx = 0; idx < boxes.size(); ++idx) {
        const auto &b = boxes[idx];
        std::array<nvtype::Float3, 8> cs;
        boxCorners(b, cs);
        const int edges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},
            {4,5},{5,6},{6,7},{7,4},
            {0,4},{1,5},{2,6},{3,7}
        };
        edge_points.clear();
        for (int e = 0; e < 12; ++e) {
            interpolateEdge(cs[edges[e][0]], cs[edges[e][1]], edge_step, edge_points);
        }
        float intensity = -2.0f;
        for (const auto &q : edge_points) {
            points.push_back({q.x, q.y, q.z, intensity});
        }
    }
}

/**
 * Save point cloud with bounding boxes and a line segment to a PCD file
 * Calls SaveBoxesAsPCD and then adds a line segment with intensity -2
 * @param boxes Vector of bounding boxes to render
 * @param points_xyzi Point cloud data as x,y,z,intensity
 * @param num_points Number of points in the point cloud
 * @param file_name Output file path
 * @param edge_step Step size for edge interpolation (default: 0.05)
 * @param line_start_x Line segment start point x coordinate
 * @param line_start_y Line segment start point y coordinate
 * @param line_start_z Line segment start point z coordinate
 * @param line_end_x Line segment end point x coordinate
 * @param line_end_y Line segment end point y coordinate
 * @param line_end_z Line segment end point z coordinate
 */
inline bool SaveBoxesAsPCDWithLine(const std::vector<detect::ProcessingBox> &boxes,
                                    const float *points_xyzi,
                                    int num_points,
                                    const std::string &file_name,
                                    float edge_step,
                                    float line_start_x, float line_start_y, float line_start_z,
                                    float line_end_x, float line_end_y, float line_end_z,
                                    std::vector<std::array<float, 4>> &out_points) {
    // First call SaveBoxesAsPCD to get boxes and point cloud
    bool result = SaveBoxesAsPCD(boxes, points_xyzi, num_points, "", edge_step, out_points);
    
    // Add line segment points with intensity -2
    nvtype::Float3 line_start(line_start_x, line_start_y, line_start_z);
    nvtype::Float3 line_end(line_end_x, line_end_y, line_end_z);
    
    std::vector<nvtype::Float4> line_points;
    interpolateEdge(line_start, line_end, edge_step, line_points);
    
    // Add line points with intensity -2
    for (const auto &p : line_points) {
        out_points.push_back({p.x, p.y, p.z, -2.0f});
    }
    
    // Save to file if file_name is provided
    if (!file_name.empty()) {
        return save_pcd_file(file_name, out_points);
    }
    return result;
}

} // namespace detect

