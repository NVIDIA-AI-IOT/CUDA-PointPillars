#ifndef __READ_CSV_H__
#define __READ_CSV_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <logger.h>

template<typename T>
inline std::vector<std::vector<T>> ReadCSV(const std::string& filename, const int& skip_rows = 1, const int& skip_cols = 0)
{
    std::vector<std::vector<T>> data; // 二维向量存储CSV数据
    try {
        std::ifstream file(filename, std::ios::in);
        LogInfo("trying to open csv file: %s", filename.c_str());
        if (!file.is_open()) {
            LogError("opening file failed: %s", filename.c_str());
            return data; // 返回空数据，表示读取失败
        }
        
        int row_cnt = 0;
        std::string line;
        while (std::getline(file, line)) {
            if (row_cnt < skip_rows) {
                row_cnt ++;
                continue; // 跳过前 skip_rows 行（标题行）
            }

            std::istringstream iss(line);
            std::string cell;
            std::vector<T> row;

            int col_cnt = 0;
            while (std::getline(iss, cell, ',')) { // 假设CSV文件以逗号分隔
                if (col_cnt < skip_cols) {
                    col_cnt ++;
                    continue; // 跳过前 skip_cols 列
                }
                row.push_back(std::stod(cell)); // 将字符串转换为double，并添加到行向量中
            }

            data.push_back(row); // 将行向量添加到二维数据向量中
        }

        file.close();
    } catch (const std::exception& e) {
        LogFatal("error reading csv file: %s", e.what());
        return data;
    }
    return data;
}

// 只读表格的第一列，因为格式已固定
template<typename T>
inline std::vector<T> ReadCSV_1D(const std::string& filename, const int& skip_rows = 1)
{
    std::vector<T> data; // 二维向量存储CSV数据
    try {
        std::ifstream file(filename, std::ios::in);
        LogInfo("trying to open csv file: %s", filename.c_str());
        if (!file.is_open()) {
            LogError("opening file failed: %s", filename.c_str());
            return data; // 返回空数据，表示读取失败
        }
        
        int row_cnt = 0;
        std::string line;
        while (std::getline(file, line)) {
            if (row_cnt < skip_rows) {
                row_cnt ++;
                continue; // 跳过前 skip_rows 行（标题行）
            }

            std::istringstream iss(line);
            std::string cell;
            std::vector<T> row;

            while (std::getline(iss, cell, ',')) { // 假设CSV文件以逗号分隔
                row.push_back(std::stod(cell)); // 将字符串转换为double，并添加到行向量中
            }

            data.push_back(row[0]); // 将第一个数据添加到数据向量中
        }

        file.close();
    } catch (const std::exception& e) {
        LogFatal("error reading csv file: %s", e.what());
        return data;
    }
    return data;
}

#endif
