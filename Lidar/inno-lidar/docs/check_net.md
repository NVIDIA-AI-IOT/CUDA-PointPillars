# check_net.sh
## A. What is check_net.sh
```shell
check_net.sh is used to test network perfrmance between client_sdk and lidar, such as network bandwidth, network jitter and the loss packet rate of get_pcd when using TCP, unicat of UDP and broadcast of UDP.
```

## B. Environment
```shell
sudo apt install iperf3
sudo apt install curl or sudo apt install wget
```

## C. Introduction of parameter
```shell
# you can use (sh check_net.sh -h) to get all information
-n|--lidar-ip                                # IP address of attached Falcon (default 172.168.1.10)
-k|--keep-inno-pc                            # Don't delete test .inno_pc files (default false)
--skip-get-pcd                               # Don't run get_pcd tests (default false)
--skip-iperf                                 # Don't run iperf tests (default false)
-v|--verbose                                 # Write more information to stdout (default false)
```

## D. How to use check_net.sh
```shell
sh check_net.sh -n 172.168.1.10
```

## F. Location
If you want to get detial of check_net.sh, please press [here](../apps/tools/check_net/check_net.sh)
