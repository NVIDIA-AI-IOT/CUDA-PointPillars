# Point Cloud Rearrangement Function Introduction

## Function Introduction

According to the position of the point cloud's horizontal angle and vertical angle, using preset horizontal angle range, vertical angle range, etc. to calculate the position of the point cloud, so that the position of the point cloud in the output array is consistent with the actual spatial position.

For example, according to the preset horizontal angle range (`azi1 ~ azi2`) and resolution (`azi_resol`), the point cloud width `W` is obtained. According to the preset vertical angle range (`ele1 ~ ele2`) and resolution (`ele_resol`), the point cloud height `H` is obtained. Then the size of the output array is `W * H`. If a point cloud has a horizontal angle of `azi` and a vertical angle of `ele`, then its offset in the output array is `(azi - azi1) / azi_resol * H + (ele - ele1) / ele_resol`.

## Parameter Introduction

Configuration parameter definitions can be found in [RemakeConfig](../libhesai/Common/include/hs_com.h)

- **`flag`**: Whether to enable rearrangement

- **`min_azi / max_azi`**: Horizontal angle range

- **`ring_azi_resolution`**: Horizontal angle resolution

- **`max_azi_scan`**: Point cloud width, (max_azi - min_azi) / ring_azi_resolution, rounded to integer

- **`min_elev / max_elev`**: Vertical angle range

- **`ring_elev_resolution`**: Vertical angle resolution

- **`max_elev_scan`**: Point cloud height, (max_elev - min_elev) / ring_elev_resolution, rounded to integer

> Note: If parameters 2~8 maintain their original configuration, default parameters corresponding to the lidar model will be used

## Additional References

Some lidars have non-uniform vertical angle resolution, meaning that in the case of one horizontal angle, there are some empty points in a column of point clouds (different vertical angle positions).

You can consider using channel numbers instead of vertical angles for rearrangement. The required modifications are as follows:

1. In the `DoRemake` function in [general_parser.cc](../libhesai/UdpParser/src/general_parser.cc), delete the calculation logic of `elev_` and directly assign the input parameter `elev` to `new_elev_iscan`.

2. Confirm the point cloud UDP version number of your lidar, which can be viewed through `wireshark` packet capture. Generally, the first two bytes in point cloud UDP packets are 0xEE 0xFF, and the following two bytes are the version number (converted to decimal). Using `Pandar128E3X` as an example, the version number is `1.4`

3. Navigate to file [udp1_4_parser.cc](../libhesai/UdpParser/src/udp1_4_parser.cc), search for the `DoRemake` function, and modify the second input parameter `elevation` to channel number `pointData.channel_index`.