# get_pcd
## A. What is get_pcd
```shell
1. Get pointcloud data and store data in different types of files, such as .inno_pc, .bag, .pcd, etc.
2. Change the type of data storage files, e.g. change .inno_pc file to .pcd file
3. split one file into several files.
```

## B. Introduction of parameter
```shell
# you can use ./get_pcd -h or ./get_pcd to get all information
--lidar-ip 172.168.1.10                                       # 172.168.1.10: lidar IP
--lidar-port 8010                                             # 8010: lidar tcp port
--lidar-udp-port 8010                                         # 8010: lidar udp port
--use-tcp                                                     # use tcp, not udp
--lidar-mode 3                                                # 3: normal mode
                                                              # 5: calibration mode
--reflectance 1                                               # 1: intensity
                                                              # 2: reflectance
--multireturn 1                                               # 1: single
                                                              # 2: twostrongest
                                                              # 3: strongest+furthest
--falcon-eye 40 10                                            # 40 10: horizontal degree ranges from -60 to 60 and vertical degree ranges from -20 to 20
--inno-pc-filename test.ino_pc                                # test.inno_pc: the name of file
--use-xyz-point 0                                             # 0: data type is ITEM_TYPE_SPHERE_POINTCLOUD or INNO_ITEM_TYPE_XYZ_POINTCLOUD and enumerate each block and channel points
                                                              # 1: data type is INNO_ITEM_TYPE_SPHERE_POINTCLOUD and covert to INNO_ITEM_TYPE_XYZ_POINTCLOUD, then enumerate each xyz points
                                                              # 2: data type is INNO_ITEM_TYPE_XYZ_POINTCLOUD and enumerate each xyz points
--roll                                                        #<ROLL_IN_DEGREES>@@@@@@@
--pitch                                                       #<PITCH_IN_DEGREES>@@@@@@@
--yaw                                                         #<YAW_IN_DEGREES>@@@@@@@@@
--file-number 10                                              # 10: the number of generated files
--frame-start 0                                               # 0: record data from Nth FRAME
--frame-number 10                                             # 10: the number of frames to be recorded
--output-filename test.pcd                                    # test.pcd: name and file type
                                                              # test.csv: name and file type
                                                              # test.inno_pc: name and file type
                                                              # test.inno_pc_xyz: name and file type
                                                              # test.inno_cframe: name and file type
                                                              # test.bag: name and file type
                                                              # test.png: name and file type
--ascii-pcd                                                   # store files in ASCII format
--extract-message                                             # get message callback information
--no-process-data                                             # do not process data, e.g.convert data into spherical coordinates
--latency-file latency.data                                   # latency.data: name of latency file
--run-time 43200                                              # 43200: run time, xx seconds
```

## C. How to use get_pcd
```shell
cd /xxxx/apps/example

#record point cloud data using udp in the form of various files
# inno_pc
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.inno_pc

# pcd(binary)
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.pcd

# pcd(ASCII)
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.pcd --ascii-pcd

# csv
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.csv

# inno_pc_xyz
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.inno_pc_xyz

# inno_cframe
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.inno_cframe

# bag
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.bag

# png
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.png

# if you want to get data by using TCP, just add --use-tcp, for example:
./get_pcd --lidar-ip 172.168.1.10 --lidar-port 8010 --lidar-udp-port 8010 --frame-number 100 --output-filename test.inno_pc --use-tcp





# change .inno_pc file to different types of files
# pcd(binary)
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number 1000 --output-filename test.pcd

# pcd(ASCII)
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number 1000 --output-filename test.pcd --ascii-pcd

# csv
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number 1000 --output-filename test.csv

# inno_pc_xyz
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number 1000 --output-filename test.inno_pc_xyz

# inno_cframe
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number 1000 --output-filename test.inno_cframe

# bag
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number 1000 --output-filename test.bag

# png
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number 1000 --output-filename test.png





# File splitting
# inno_pc
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.inno_pc

# pcd(binary)
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.pcd

# pcd(ASCII)
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.pcd --ascii-pcd

# csv
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.csv

# inno_pc_xyz
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.inno_pc_xyz

# inno_cframe
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.inno_cframe

# bag
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.bag

# png
./get_pcd --inno-pc-filename input.inno_pc --frame-start 0 --frame-number m --file-number n --output-filename test.png
```

## D. Location
If you want to get detial of get_pcd, please press [here](../apps/tools/get_pcd/get_pcd.cpp)
