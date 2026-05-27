#pragma once
#include <iostream>
#include <string>
#include <string.h>
#include <cstdint>

namespace hesai
{
namespace lidar
{

typedef struct TransformParam  
{
  bool use_flag = true;
  ///< unit, m
  float x = 0.0f; 
  ///< unit, m     
  float y = 0.0f;
  ///< unit, m      
  float z = 0.0f;  
  ///< unit, radian    
  float roll = 0.0f;  
  ///< unit, radian 
  float pitch = 0.0f;
  ///< unit, radian
  float yaw = 0.0f;    
} TransformParam;

struct RemakeConfig {
  bool flag = false;
  float min_azi = -1;
  float max_azi = -1;
  float ring_azi_resolution = -1;
  int max_azi_scan = -1;   // (max_azi - min_azi) / ring_azi_resolution
  float min_elev = -1;
  float max_elev = -1;
  float ring_elev_resolution = -1;
  int max_elev_scan = -1;   // (max_elev - min_elev) / ring_elev_resolution
};


}
}