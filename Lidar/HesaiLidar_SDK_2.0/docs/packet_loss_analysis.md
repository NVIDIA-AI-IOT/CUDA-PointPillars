# Packet Loss Statistics
The packet loss statistics function implements real-time packet loss statistics for Hesai lidar UDP data, based on packet loss detection and statistics using the sequence number (UDP Sequence) field in each UDP point cloud data packet.


## Preparation
There are three methods to count the packet loss rate of lidar UDP data parsed by the SDK. This shows **Method 1**. Other methods are referenced in **Additional References**.

#### Method 1: Count packet loss rate and timestamp jump rate within a time period
Navigate to [packet_loss_tool.cc](../tool/packet_loss_tool.cc)

For parsing configuration reference, see **[How to Parse Lidar Data Online](../docs/parsing_lidar_data_online.md)** and **[How to Parse PCAP File Data Offline](../docs/parsing_pcap_file_data_offline.md)**

You can modify the packet loss statistics time in the code as needed:
```cpp
float run_time = 15;  // Statistics for 15s
```

## Steps
### 1 Compilation
In the HesaiLidar_SDK_2.0 folder, open a terminal and execute the following commands:
```bash
cd HesaiLidar_SDK_2.0/tool
mkdir build
cd build
cmake ..
make
```

### 2 Run
After successful compilation, run the generated packet_loss_tool executable file in the build folder. You can add parameters to specify the runtime (s). If no runtime is specified, the default is 15s:
```bash
./packet_loss_tool 15
```
Output example:
```log
total recevice packet time: 15000ms  // Statistics time
total receviced packet count: 93229  // Total packet count
package loss: 
total loss packet count: 0  // Total lost packet count  
timestamp loss: 
total loss packet count: 0  // Total timestamp jump count
```


## Additional References
#### Method 2: Count total packet loss rate between two packet losses within 1s
Navigate to [test.cc](../test/test.cc)
```cpp
// Enable packet loss statistics
param.decoder_param.enable_packet_loss_tool = true; // Enable packet loss statistics function
```
Compilation and execution methods refer to Method 1. When packet loss occurs, the terminal will print warning information similar to the following:
```log
[WARNING] pkt loss freq: 3 / 56268
```
This indicates that within the 1s statistical period, from the last packet loss to the next new packet loss, a total of 3 packets were lost, and the total number of packets that should have been received was 56268.

#### Method 3: Users add functionality to count packet loss rate themselves
According to user requirements, callback functions can be added to output current frame packet loss rate and cumulative packet loss rate in the terminal. For example, add the following callback function in [test.cc](../test/test.cc):
```cpp
void packetLossCallback(const uint32_t& total_packets, const uint32_t& lost_packets) {
  static uint32_t last_total = 0;
  static uint32_t last_loss = 0;

  // Compute packet count and loss since last frame
  uint32_t frame_total = total_packets - last_total;
  uint32_t frame_loss  = lost_packets - last_loss;

  // Update static tracking variables
  last_total = total_packets;
  last_loss  = lost_packets;

  // Calculate loss rates
  double frame_loss_rate = frame_total == 0 ? 0.0 :
                           static_cast<double>(frame_loss) / frame_total * 100.0;
  double total_loss_rate = total_packets == 0 ? 0.0 :
                           static_cast<double>(lost_packets) / total_packets * 100.0;

  // Print results
  printf("[Frame Loss Rate]  %.2f%% (%u / %u)\n", frame_loss_rate, frame_loss, frame_total);
  printf("[Total Loss Rate]  %.2f%% (%u / %u)\n", total_loss_rate, lost_packets, total_packets);
}
```
And call this callback function:
```cpp
sample.RegRecvCallback(packetLossCallback);
```
Compilation and execution methods refer to **Method 1**.

Output example:
```log
[Frame Loss Rate]  97.47% (20241 / 20767)
[Total Loss Rate]  2.12% (20241 / 956569)
```
