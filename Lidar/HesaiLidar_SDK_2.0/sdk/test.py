import hsai_sdk_wrapper_python as hsai
import time
import threading

exit_flag = threading.Event()

def start_lidar():
    try:
        hsai.init_hsai_lidar_sdk_cplusplus_interface()
        print("hsai SDK started.")
    except Exception as e:
        print(f"hsai error: {e}")
        exit_flag.set()

def wait_for_exit():
    while not exit_flag.is_set():
        user_input = input("Type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            exit_flag.set()

def main():
    print("begin *******")
    lidar_thread = threading.Thread(target=start_lidar)
    input_thread = threading.Thread(target=wait_for_exit)

    lidar_thread.start()
    input_thread.start()

    while not exit_flag.is_set():
        pc = hsai.get_hsai_lidar_pointcloud_data_interface(160)
        if pc:
            points = pc['points']

            for i in range(min(5, points.shape[0])):
                x, y, z = points[i]
                print(f"python-----------------[Main] Point {i}: x={x:.3f}, y={y:.3f}, z={z:.3f} ")
        else:
            time.sleep(0.1)

    print("Shutting down hsai SDK...")
    hsai.uninit_hsai_lidar_sdk_cplusplus_interface()  
    lidar_thread.join()
    print("Shutdown complete.")

if __name__ == "__main__":
    main()
