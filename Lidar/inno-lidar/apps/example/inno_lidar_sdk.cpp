#if defined(__MINGW64__) || !defined(_WIN32)
#include <getopt.h>
#include <unistd.h>
#else
#include "src/utils/getopt_windows.h"
#endif

#include <fcntl.h>

#include <string>
#include <thread>
#include <vector>

#include "src/sdk_common/inno_lidar_api.h"
#include "src/sdk_common/inno_lidar_other_api.h"
#include "src/sdk_common/inno_lidar_packet_utils.h"
#include "src/utils/inno_lidar_log.h"
#if !defined(__MINGW64__) && defined(_WIN32)
#include <io.h>
#include <stdarg.h>
#include <stdio.h>
#endif
#include <iostream>
#include <fstream>
#include <iomanip>  // for std::setprecision
#include <sstream>

#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
namespace py = pybind11;

static size_t kmaxPointNumberOneFrame = 500000;
static const double kUsInSecond = 1000.0;
static const double k10UsInSecond = 100.0;
struct LidarOption {
  std::string lidar_ip = "172.168.1.10";
  uint16_t lidar_port = 8010;
  uint16_t lidar_udp_port = 8010;
};

struct TransformParam
{
  bool use_flag = false;
  float roll =0.0;
  float pitch =0.0;
  float yaw =0.0;
  float x=0.0;
  float y=0.0;
  float z=0.0;
} transform_global ;

void TransformPoint(float& x, float& y, float& z, const TransformParam& transform)
{
  if (transform.use_flag == false) return;

  float cosa = std::cos(transform.roll);
  float sina = std::sin(transform.roll);
  float cosb = std::cos(transform.pitch);
  float sinb = std::sin(transform.pitch);
  float cosc = std::cos(transform.yaw);
  float sinc = std::sin(transform.yaw);

  float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
              (sina * sinc + cosa * sinb * cosc) * z + transform.x;
  float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
              (cosa * sinb * sinc - sina * cosc) * z + transform.y;
  float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;
  x = x_;
  y = y_;
  z = z_; 
}


struct PointData {
    float x, y, z;
    uint16_t reflectivity;
    uint8_t tag;
    uint64_t timestamp;
};
std::mutex queue_mutex;
std::vector<PointData> pointcloud_queue;
std::thread inno_thread;
bool sdk_running;

class CallbackProcessor {
 public:
  explicit CallbackProcessor();

  void set_done() { done_ = true; }
  bool is_done() { 
    if(sdk_running==false)
      done_= false;
    return done_; }

  /*@param context Callback context passed in inno_lidar_set_callbacks().
   * @param from_remote > 0 means the message is from remote source
   * @param level Severity of message.
   * @param code Error code.
   * @param error_message Error message.
   * @return void.
   */
  void process_message(const enum InnoMessageLevel level, const enum InnoMessageCode code, const char *error_message);

  /*
   * @param context Callback context passed in inno_lidar_set_callbacks().
   * @param data Pointer to InnoDataPacket
   * @return 0
   */
  int process_data(int handler, const InnoDataPacket &pkt);

  /*
   * @param context Callback context passed in inno_lidar_set_callbacks().
   * @param status Pointer to InnoDataPacket
   * @return 0
   */
  int process_status(const InnoStatusPacket &pkt);

   /*
    recorder lidar send inno_pc data 
   */
  int recorder_data(int lidar_handle, void *context, enum InnoRecorderCallbackType type,
                                      const char *buffer, int len);

 private:
  CallbackProcessor(const CallbackProcessor &) = delete;
  CallbackProcessor &operator=(const CallbackProcessor &) = delete;

 private:
  // Current frame idx
  int64_t current_frame_ = -1;
  // The number of frames received so far
  int64_t frame_so_far_ = -1;
  // Application stop
  volatile bool done_ = false;
  // Store one frame of pcd point
  std::vector<PointData> frame_data_;
};

void save_frame_to_pcd(const std::vector<PointData> &points, int64_t frame_id) {
  std::cout <<"save_frame_to_pcd \n";
  if (points.empty()) {
    std::cout << "Frame " << frame_id << " is empty, skip saving.\n";
    return;
  }

  // 文件名：frame_00001.pcd
  std::ostringstream oss;
  oss << "frame_" << std::setw(5) << std::setfill('0') << frame_id << ".pcd";
  std::string filename = oss.str();

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  // 写入 PCD �?
  ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z intensity\n";
  ofs << "SIZE 4 4 4 4\n";
  ofs << "TYPE F F F F\n";
  ofs << "COUNT 1 1 1 1\n";
  ofs << "WIDTH " << points.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << points.size() << "\n";
  ofs << "DATA ascii\n";

  // 写入每个点的数据
  ofs << std::fixed << std::setprecision(4);
  for (const auto &p : points) {
    ofs << p.x << " " << p.y << " " << p.z << " " << static_cast<float>(p.reflectivity) << "\n";
  }

  ofs.close();
  std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
}

CallbackProcessor::CallbackProcessor() {
  // reserve memory for store one frame data
  frame_data_.reserve(kmaxPointNumberOneFrame);
}

void CallbackProcessor::process_message(const enum InnoMessageLevel level, const enum InnoMessageCode code,
                                        const char *error_message) {

  // process exception
  if ((level == INNO_MESSAGE_LEVEL_INFO && code == INNO_MESSAGE_CODE_READ_FILE_END) ||
      (level == INNO_MESSAGE_LEVEL_CRITICAL && code == INNO_MESSAGE_CODE_CANNOT_READ)) {
    this->set_done();
  }

  switch (level) {
    case INNO_MESSAGE_LEVEL_INFO:
      //printf("info_content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_WARNING:
      //printf("waring_content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_ERROR:
      printf("error_content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_FATAL:
      printf("error_content = %s\n", error_message);
      break;
    case INNO_MESSAGE_LEVEL_CRITICAL:
      printf("error_content = %s\n", error_message);
      break;
    default:
      printf("error_content = %s\n", error_message);
      break;
  }
}

int CallbackProcessor::process_status(const InnoStatusPacket &pkt) {
  // sanity check
  if (!inno_lidar_check_status_packet(&pkt, 0)) {
    printf("corrupted pkt->idx = %" PRI_SIZEU, pkt.idx);
    return 0;
  }

  static size_t cnt = 0;
  if (cnt++ % 100 == 1) {
    constexpr size_t buf_size = 2048;
    char buf[buf_size]{0};

    // int ret = inno_lidar_printf_status_packet(&pkt, buf, buf_size);
    // if (ret > 0) {
    //   printf("Received status packet #%lu : %s", cnt, buf);
    // } else {
    //   printf("Received status packet #%lu : errorno: %d", cnt, ret);
    // }
  }

  return 0;
}

int CallbackProcessor::process_data(int handler, const InnoDataPacket &pkt) {
  // iterate each point in the pkt
  for (int i = 0; i < pkt.item_number; i++) {

      if (current_frame_ != pkt.idx) {
        // deliver the frame data to user and then clear the buffer
        std::lock_guard<std::mutex> lock(queue_mutex);
        pointcloud_queue.clear();
        //save_frame_to_pcd(frame_data_,current_frame_) ;

        for(int i =0 ; i<frame_data_.size() ;i++){
          TransformPoint(frame_data_[i].x, frame_data_[i].y, frame_data_[i].z ,transform_global);
        }
        pointcloud_queue=std::move(frame_data_);
        frame_data_.clear();
        current_frame_ = pkt.idx;
      }
      //InnoXyzPoint *point = reinterpret_cast<InnoXyzPoint *>(const_cast<char *>(pkt.payload));
      InnoEnXyzPoint *point = reinterpret_cast<InnoEnXyzPoint *>(const_cast<char *>(pkt.payload));
      // if want to get more information, please define your pcd_point struct and copy InnoXyzPoint data to it
      if(point[i].x==0.0 && point[i].y==0.0 && point[i].z == 0.0)
        continue;
      uint64_t timestamp = pkt.common.ts_start_us / kUsInSecond + point[i].ts_10us / k10UsInSecond;
      //printf("%.3f %.3f %.3f %lld\n",point[i].x,point[i].y,point[i].z, timestamp);
      frame_data_.push_back(PointData{
          point[i].x, point[i].y, point[i].z,
          point[i].reflectance,
          (uint8_t)pkt.idx,
          timestamp
      });
  }

  return 0;
}

int CallbackProcessor::recorder_data(int lidar_handle, void *context, enum InnoRecorderCallbackType type,
                                    const char *buffer, int len) {
  InnoDataPacket *pkt = reinterpret_cast<InnoDataPacket *>(const_cast<char *>(buffer));
  printf("recorder_data type %d len %d sub_seq %u", type, len, pkt->sub_seq);
  if (CHECK_CO_SPHERE_POINTCLOUD_DATA(pkt->type)) {
    // CO_SPHERE_POINTCLOUD_DATA Need write anghv table to the file header
  }
  return 0;
}


int sdk_main_loop(const std::string &lidar_ip_addrwss ,const float roll ,const float pitch,const float yaw ,
  const float x ,const float y,const float z )
 {
  printf(" * inno_api_version %s", inno_api_version());
  printf(" * roll:%.4f\n",roll );
  printf(" * pitch:%.4f\n",pitch );
  printf(" * yaw:%.4f\n",yaw );
  transform_global.use_flag= true ;
  transform_global.roll = roll;
  transform_global.pitch = pitch;
  transform_global.yaw = yaw;
  transform_global.x = x;
  transform_global.y = y;
  transform_global.z = z;

  CallbackProcessor processor;

  // 打开录制文件（修改这里）
  //const char *pc_file = "/home/zser/inno-lidar/20251014150932_metaview.inno_pc";
  //int handle = inno_lidar_open_file("file_playback", pc_file,false,1,1,0);
  //printf("cannot open inno_pc file %s", pc_file);

  int handle = inno_lidar_open_live("live",  // name of lidar instance
                                lidar_ip_addrwss.c_str(), 8010, INNO_LIDAR_PROTOCOL_PCS_UDP,
                                8010);



  // 设置为XYZ点云输出
  inno_lidar_set_attribute_string(handle, "force_xyz_pointcloud", "1");

  // 注册回调函数
  int ret = inno_lidar_set_callbacks(
      handle,
      [](const int lidar_handle, void *ctx, const uint32_t from_remote, const enum InnoMessageLevel level,
         const enum InnoMessageCode code, const char *error_message) {
        return reinterpret_cast<CallbackProcessor *>(ctx)->process_message(level, code, error_message);
      },
      [](const int lidar_handle, void *ctx, const InnoDataPacket *pkt) -> int {
        return reinterpret_cast<CallbackProcessor *>(ctx)->process_data(lidar_handle, *pkt);
      },
      [](const int lidar_handle, void *ctx, const InnoStatusPacket *pkt) -> int {
        return reinterpret_cast<CallbackProcessor *>(ctx)->process_status(*pkt);
      },
      NULL, &processor);

  sdk_running = true;
  // 开始播放文�?
  ret = inno_lidar_start(handle);

  while (!processor.is_done()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  inno_lidar_stop(handle);
  inno_lidar_close(handle);
  return 0;
}

void init_inno_lidar_sdk_cplusplus_interface(const std::string &lidar_ip_addrwss ,const float roll ,const float pitch,const float yaw
,const float x ,const float y,const float z ) 
{ //const std::string &config_path
  // 启动 SDK 线程，不阻塞 Python
  printf(" * lidar_ip_addrwss:%s\n",lidar_ip_addrwss.c_str() );
  inno_thread = std::thread(sdk_main_loop, lidar_ip_addrwss,roll,pitch,yaw,x,y,z); //config_path
}

void uninit_inno_lidar_sdk_cplusplus_interface() {
  sdk_running = false;
  if (inno_thread.joinable())
    inno_thread.join();
}

py::dict get_inno_lidar_pointcloud_data_interface(int64_t time_interval_ms) { 
    std::vector<PointData> frame;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        std::vector<PointData> frame_temp = pointcloud_queue;
        pointcloud_queue.clear();
        frame.insert(frame.end(), 
           std::make_move_iterator(frame_temp.begin()),
           std::make_move_iterator(frame_temp.end()));
    }

    size_t n = frame.size();
    //printf("n= %d\n", n );
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(n), 3};
    py::array_t<float> points(shape);
    py::array_t<uint8_t> reflectivity(n);
    //py::array_t<uint8_t> tag(n);
    //uint64_t timestamp = frame.empty() ? 0 : frame[0].timestamp;

    auto buf_p = points.mutable_unchecked<2>();
    auto buf_r = reflectivity.mutable_unchecked<1>();
    //auto buf_t = tag.mutable_unchecked<1>();

    for (size_t i = 0; i < n; ++i) {
        buf_p(i, 0) = frame[i].x;
        buf_p(i, 1) = frame[i].y;
        buf_p(i, 2) = frame[i].z;
        buf_r(i) = frame[i].reflectivity;
        //buf_t(i) = frame[i].tag;
    }

    py::dict result;
    result["points"] = points;
    result["reflectivity"] = reflectivity;
    // result["tag"] = tag;
    // result["timestamp"] = timestamp;
    return result;
}


PYBIND11_MODULE(inno_sdk_wrapper_python, m) {
    m.def("init_inno_lidar_sdk_cplusplus_interface", &init_inno_lidar_sdk_cplusplus_interface, "Start the inno SDK in a background thread");
    m.def("uninit_inno_lidar_sdk_cplusplus_interface", &uninit_inno_lidar_sdk_cplusplus_interface, "Stop the inno SDK cleanly");
    m.def("get_inno_lidar_pointcloud_data_interface", &get_inno_lidar_pointcloud_data_interface, "Poll get_inno_lidar_pointcloud_data_interface");
}

