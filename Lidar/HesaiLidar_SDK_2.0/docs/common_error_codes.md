# Common Error Codes
Here are some common warning codes that you may encounter.

##  Troubleshooting
1. `Failed to obtain correction file from lidar!`

   This indicates that the program attempted to retrieve the angle correction data directly from the lidar but failed. After that, it still tries to load the correction file from a local path specified in the configuration `param.input_param.correction_file_path`. If a valid local file is found, the program can still proceed.

2. `Packet with invaild delimiter`

   This indicates that the program has received an unexpected data packet. If this occurs occasionally, it generally does not affect parsing and simply means that some unrecognized packets were received. However, if the message is printed continuously, it suggests that the system is consistently receiving unexpected packet data. This may be due to the LiDAR model currently in use not being supported for parsing, or the LiDAR only sending fault messages without sending point cloud data, or other related issues.
   Make sure that the data structure of lidar  is supported for parsing, and ensure that the lidar has been correctly configured.

3. `Firetimes laserId is invalid`

   This indicates that the program read an incorrect firetime correction file.

4. `Open firetime file failed`

   This indicates that the program failed to read the firetime file, but it will not prevent the program from running.