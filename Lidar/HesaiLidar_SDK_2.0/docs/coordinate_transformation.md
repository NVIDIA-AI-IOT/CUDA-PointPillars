# Coordinate Transformation
Coordinate transformation is commonly applied in extrinsic calibration and similar operations.

## Code Structure
The coordinate transformation is defined in [hs_com.h](../libhesai/Common/include/hs_com.h).
```cpp
typedef struct TransformParam  
{
  bool use_flag = false;
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
```
## Configuration
The transformation default value is given as zero.

- `use_flag` indicates whether to enable coordinate transformation

- `x,y,z` are given in `meter`

- `Roll, Pitch, Yaw` angles are given in `radian`

- The order of angles applied to the sensor is `Yaw - Pitch - Roll`

Please refer to the source code for more details.
