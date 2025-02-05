# 360环视系统

使用4个1080p，FOV190度的鱼眼相机
相机参数标定直接使用 https://github.com/neozhaoliang/surround-view-system-introduction
C++的实时画面开发基于 https://github.com/JokerEyeAdas/AdasSourrondView
在Jetson Orin AGX上运行速度30+FPS

### 上手指南

###### 配置要求

1. 系统 Ubuntu20.04
2. 架构 x86和Jetson平台均已测试通过
3. OpenCV >= 4.5.4 (with CUDA)
4. CUDA >= 11.4
5. CMake

###### **安装步骤**

```sh
git clone https://github.com/FunnyWii/Surround-View-System
mkdir build
cd build
cmake ..
make
./avm_cam
```

### 文件目录说明

```
.
├── 1.mp4                // Video Demo
├── avm_app_demo.cpp     
├── avm_cali_demo.cpp
├── avm_cam_demo.cpp    // main function
├── cache
├── calibration@neozhaoliang  // Calib camera and calculate params 
├── CMakeLists.txt      
├── doc            
├── images
├── srcs                // source and headers
└── yaml                // Yaml files
```

