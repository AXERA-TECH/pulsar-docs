.. _model_deploy_advanced:

=========================
模型部署进阶指南
=========================

--------------------
概述
--------------------

本节将提供运行 ``Pulsar`` 编译生成的 ``joint`` 模型的代码示例, 所有示例代码由 ``ax-samples`` 项目提供。
``ax-samples`` 由 AXera 主导的开源项目，其目的是提供业界主流的开源算法模型部署示例代码，方便社区开发者快速对 AXera 的芯片进行评估和适配。

~~~~~~~~~~~~~~~~~~~~
获取方式
~~~~~~~~~~~~~~~~~~~~

- `Github 版本 <https://github.com/AXERA-TECH/ax-samples>`__

.. hint::

    离线版本是随着本文档发布时从 GitHub 上获取的，存在一定的延时，若想体验最新功能，请选择 GitHub 版本。

~~~~~~~~~~~~~~~~~~~~
ax-samples 简介
~~~~~~~~~~~~~~~~~~~~

当前 ``ax-samples`` 已验证但不限于以下开源模型:

- 分类模型

  - SqueezeNetv1.1
  - MobileNetv1
  - MobileNetv2
  - ResNet18
  - ResNet50
  - VGG16
  - Others......

- 检测模型

  - PP-YOLOv3
  - YOLOv3
  - YOLOv3-Tiny
  - YOLOv4
  - YOLOv4-Tiny
  - YOLOv5m
  - YOLOv5s
  - YOLOv7-Tiny
  - YOLOX-S
  - YOLO-Fastest-XL

- 人型检测
  
  - YOLO-Fastest-Body
  
- 人脸检测
  
  - scrfd
  
- 障碍物检测 (扫地机场景)
  
  - Robot-Obstacle-Detect
  
- 3D单目车辆检测
  
  - Monodlex
  
- 人体关键点

  - HRNet
  
- 人体分割
  
  - PP-HumanSeg
  
- 语义分割

  - PP-Seg

- 姿态模型

  - HRNet
  
已验证硬件平台

- AX630A
- AX620A/U

``ax-sampless`` 目录说明

.. code-block:: bash

    $ tree -L 2
    .
    ├── CMakeLists.txt
    ├── LICENSE
    ├── README.md
    ├── README_EN.md
    ├── benchmark
    │   └── README.md
    ├── cmake
    │   ├── check.cmake
    │   └── summary.cmake
    ├── docs
    │   ├── AX620A.md
    │   ├── AX620U.md
    │   ├── body_seg_bg_res.jpg
    │   ├── compile.md
    │   ├── seg_res.jpg
    │   └── yolov3_paddle.jpg
    ├── examples
    │   ├── CMakeLists.txt
    │   ├── README.md
    │   ├── ax_classification_accuracy.cc
    │   ├── ax_classification_nv12_resize_steps.cc
    │   ├── ax_classification_steps.cc
    │   ├── ax_crop_resize_nv12.cc
    │   ├── ax_hrnet_steps.cc
    │   ├── ax_ld_model_mmap.cc
    │   ├── ax_models_load_inspect.cc
    │   ├── ax_monodlex_steps.cc
    │   ├── ax_nanodet_steps.cc
    │   ├── ax_paddle_mobilehumseg_steps.cc
    │   ├── ax_paddle_mobileseg.cc
    │   ├── ax_paddle_yolov3_steps.cc
    │   ├── ax_robot_obstacle_detect_steps.cc
    │   ├── ax_scrfd_steps.cc
    │   ├── ax_yolo_fastest_body_steps.cc
    │   ├── ax_yolo_fastest_steps.cc
    │   ├── ax_yolov3_accuracy.cc
    │   ├── ax_yolov3_steps.cc
    │   ├── ax_yolov3_tiny_steps.cc
    │   ├── ax_yolov4_steps.cc
    │   ├── ax_yolov4_tiny_3l_steps.cc
    │   ├── ax_yolov4_tiny_steps.cc
    │   ├── ax_yolov5s_620u_steps.cc
    │   ├── ax_yolov5s_steps.cc
    │   ├── ax_yolov7_steps.cc
    │   ├── ax_yoloxs_steps.cc
    │   ├── base
    │   ├── cv
    │   ├── middleware
    │   └── utilities
    └── toolchains
        ├── aarch64-linux-gnu.toolchain.cmake
        └── arm-linux-gnueabihf.toolchain.cmake

以上目录包含了用于演示的控制台 ``Demo``. 在 ``Linux`` 系统下, 通过控制台运行.

--------------------
编译示例
--------------------

**ax-samples** 的源码编译目前有两种实现路径：

- 基于 AX-Pi 的本地编译，因为 AX-Pi 上集成的完成了软件开发环境，操作简单；
- 嵌入式 Linux 交叉编译。

~~~~~~~~~~~~~~~~~~~~
环境准备
~~~~~~~~~~~~~~~~~~~~

- ``cmake`` 版本大于等于 ``3.13``
- ``AX620A`` 配套的交叉编译工具链 ``arm-linux-gnueabihf-gxx`` 已添加到环境变量中

^^^^^^^^^^^^^^^^^^^^
安装 cmake
^^^^^^^^^^^^^^^^^^^^

``cmake`` 的安装有多种方式, 如果是 ``Anaconda`` **虚拟环境** 下, 可以通过如下命令安装:

.. code-block:: bash
  
  pip install cmake

如果 **非虚拟环境** , 且系统为 ``Ubuntu``, 可以通过

.. code-block:: bash

  sudo apt-get install cmake

.. _`cmake 官网`: https://cmake.org/download/

如果安装版本较低, 也可以通过下载 **源码编译** ``cmake``, 具体方法如下:

- step 1: `cmake 官网`_ 下载 ``cmake`` 后解压

- step 2: 进入安装文件夹, 依次执行

  .. code-block:: bash
    
    ./configure
    make -j4  # 4代表电脑核数, 可以省略
    sudo make install

- step 3: ``cmake`` 安装完毕后, 通过以下命令查看版本信息

  .. code-block:: bash

    cmake --version

.. _`arm-linux-gnueabihf-gxx`: http://releases.linaro.org/components/toolchain/binaries/latest-7/arm-linux-gnueabihf/

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
安装交叉编译工具 arm-linux-gnueabihf-gxx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

交叉编译器有很多种, 推荐使用 ``Linaro`` 出品的交叉编译器, 可以从 `arm-linux-gnueabihf-gxx`_ 中下载相关文件, 
其中 ``gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz`` 为 64bit 版本.

.. code-block:: bash

  # 新建文件夹并移动压缩包
  mkdir -p ~/usr/local/lib
  mv gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar ~/usr/local/lib

  # 解压
  xz -d gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar.xz
  tar -xvf gcc-linaro-7.5.0-2019.12-x86_64_arm-linux-gnueabihf.tar

  # 配置环境变量
  vim ~/.bashrc
  export PATH=$PATH:~/usr/local/lib/gcc-linaro-x86_64_arm-linux-gnueabihf/bin

  # 环境生效
  source ~/.bashrc

~~~~~~~~~~~~~~~~~~~~
交叉编译
~~~~~~~~~~~~~~~~~~~~

**下载源码**

.. code-block:: bash

    git clone https://github.com/AXERA-TECH/ax-samples.git


**3rdparty 目录准备**

.. _`AX620A/U 对应版本`: https://github.com/AXERA-TECH/ax-samples/releases/download/v0.1/opencv-arm-linux-gnueabihf-gcc-7.5.0.zip
.. _`AX630A 对应版本`: https://github.com/AXERA-TECH/ax-samples/releases/download/v0.1/opencv-aarch64-linux-gnu-gcc-7.5.0.zip

- 下载预编译好的 OpenCV 库文件 

    - `AX620A/U 对应版本`_

    - `AX630A 对应版本`_

- 在 ax-samples 创建 3rdparty 文件，并将下载好的 OpenCV 库文件压缩包解压到该文件夹中。

**BSP 依赖库准备**

获取 AX620 BSP 开发包后，执行如下操作

.. code-block:: bash

    tar -zxvf AX620_SDK_XXX.tgz
    cd AX620_SDK_XXX/package
    tar -zxvf msp.tgz

**源码编译**

进入 ax-samples 根目录，创建 cmake 编译任务：

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake -DBSP_MSP_DIR=${AX620_SDK_XXX}/msp/out/ ..
    $ make install


编译完成后，生成的可执行示例存放在 `ax-samples/build/install/bin/` 路径下：

.. code-block:: bash

    ax-samples/build$ tree install
    install
    └── bin
        ├── ax_classification
        ├── ax_classification_accuracy
        ├── ax_classification_nv12
        ├── ax_cv_test
        ├── ax_hrnet
        ├── ax_models_load_inspect
        ├── ax_monodlex
        ├── ax_nanodet
        ├── ax_paddle_mobilehumseg
        ├── ax_paddle_mobileseg
        ├── ax_paddle_yolov3
        ├── ax_robot_obstacle
        ├── ax_scrfd
        ├── ax_yolo_fastest
        ├── ax_yolo_fastest_body
        ├── ax_yolov3
        ├── ax_yolov3_accuracy
        ├── ax_yolov3_tiny
        ├── ax_yolov4
        ├── ax_yolov4_tiny
        ├── ax_yolov4_tiny_3l
        ├── ax_yolov5s
        ├── ax_yolov5s_620u
        ├── ax_yolov7
        └── ax_yoloxs

~~~~~~~~~~~~~~~~~~~~
本地编译
~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^
硬件需求
^^^^^^^^^^^^^^^^^^^^

- AX-Pi（基于 AX620A，面向社区开发者的高性价比开发板）

^^^^^^^^^^^^^^^^^^^^
编译过程
^^^^^^^^^^^^^^^^^^^^

git clone 下载源码，进入 ``ax-samples`` 根目录，创建 ``cmake`` 编译任务：

.. code-block:: bash

  $ git clone https://github.com/AXERA-TECH/ax-samples.git
  $ cd ax-samples
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make install

编译完成后，生成的可执行示例存放在 ``ax-samples/build/install/bin/`` 路径下：

.. code-block:: bash

  ax-samples/build$ tree install
  install
  └── bin
      ├── ax_classification
      ├── ax_classification_accuracy
      ├── ax_classification_nv12
      ├── ax_cv_test
      ├── ax_hrnet
      ├── ax_models_load_inspect
      ├── ax_monodlex
      ├── ax_nanodet
      ├── ax_paddle_mobilehumseg
      ├── ax_paddle_mobileseg
      ├── ax_paddle_yolov3
      ├── ax_robot_obstacle
      ├── ax_scrfd
      ├── ax_yolo_fastest
      ├── ax_yolo_fastest_body
      ├── ax_yolov3
      ├── ax_yolov3_accuracy
      ├── ax_yolov3_tiny
      ├── ax_yolov4
      ├── ax_yolov4_tiny
      ├── ax_yolov4_tiny_3l
      ├── ax_yolov5s
      ├── ax_yolov5s_620u
      ├── ax_yolov7
      └── ax_yoloxs  


--------------------
运行示例
--------------------

**运行准备**

.. warning::

  这一节的示例只有 ``ax-samples`` , 并没有提供 ``mobilenetv2`` 和 ``yolov5s`` 的任何模型, 以下 log 仅供参考.

登入 ``AX620A`` 开发板, 在 ``root`` 路径下创建 ``samples`` 文件夹. 

- 将 ``build/install/bin/`` 中编译生成的可执行示例拷贝到 ``/root/ax-samples/`` 路径下;
- 将 **Pulsar** 生成的 ``mobilenetv2.joint`` 或 ``yolov5s.joint`` 模型拷贝到  ``/root/ax-samples/`` 路径下;
- 将测试图片拷贝到 ``/root/ax-samples/`` 路径下.

.. attention::

  注意: 示例代码中并未提供 ``mobilenetv2.joint`` 等检测模型, 需要自行从开源 ``onnx`` 模型进行转换.

.. code-block:: bash
  
  /root/ax-samples # ls -l
  total 40644
  -rwx--x--x    1 root     root       3805332 Mar 22 14:01 ax_classification
  -rwx--x--x    1 root     root       3979652 Mar 22 14:01 ax_yolov5s
  -rw-------    1 root     root        140391 Mar 22 10:39 cat.jpg
  -rw-------    1 root     root        163759 Mar 22 14:01 dog.jpg
  -rw-------    1 root     root       4299243 Mar 22 14:00 mobilenetv2.joint
  -rw-------    1 root     root      29217004 Mar 22 14:04 yolov5s.joint

如果提示板子空间不足, 可以通过文件夹挂载的方式解决.

**MacOS 挂载 ARM 开发板示例**

.. hint::

  由于板上空间有限, 测试时通常需要进行文件夹共享操作, 这个时候就需要将 ``ARM`` 开发板与主机之间进行共享. 这里仅以 ``MacOS`` 为例.

开发机挂载 ``ARM`` 开发板需要 ``NFS`` 服务, 而 ``MacOS`` 系统自带 ``NFS`` 服务, 只需要创建 ``/etc/exports`` 文件夹, ``nfsd`` 将自动启动并开始用于 ``exports``.

``/etc/exports`` 可以配置如下:

.. code-block:: shell

  /path/your/sharing/directory -alldirs -maproot=root:wheel -rw -network xxx.xxx.xxx.xxx -mask 255.255.255.0

参数释义

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - 参数名
      - 含义
    * - alldirs
      - 共享 ``/Users`` 目录下所有文件, 如果只想共享一个文件夹可以省略
    * - network
      - 挂载 ARM 开发板 IP 地址, 可以是网段地址
    * - mask
      - 子网掩码, 通常是 255.255.255.0
    * - maproot
      - 映射规则, 当 ``maproot=root:wheel`` 时表示把 ``ARM`` 板的 ``root`` 用户映射为开发机上的 ``root`` 用户, ``ARM`` 的 ``root`` 组 映射为 ``MacOS`` 上的 ``wheel`` (gid=0) 组. 
        如果缺省, 可能会出现 ``nfsroot`` 链接失败错误.
    * - rw
      - 读写操作, 默认开启

修改 ``/etc/exports`` 需要重启 ``nfsd`` 服务

.. code-block:: bash

  sudo nfsd restart

如果配置成功, 可以使用

.. code-block:: bash

  sudo showmount -e
 
命令查看挂载信息, 例如输出 ``/Users/skylake/board_nfs 10.168.21.xx``, 配置好开发机后需要在 ``ARM`` 端执行 ``mount`` 指令

.. code-block:: bash

  mount -t nfs -o nolock,tcp macos_ip:/your/shared/directory /mnt/directory

如果出现权限问题, 需要检查 ``maproot`` 参数是否正确.

.. hint::

  ``network`` 参数可以配置成网段的形式, 如: ``10.168.21.0``, 如果挂载单ip出现 ``Permission denied``, 可以尝试一下网段内挂载.

**分类模型**

对于分类模型, 可以通过执行 ``ax_classification`` 程序实现板上运行.

.. code-block:: bash

  /root/ax-samples # ./ax_classification -m mobilenetv2.joint -i cat.jpg -r 100
  --------------------------------------
  model file : mobilenetv2.joint
  image file : cat.jpg
  img_h, img_w : 224 224
  Run-Joint Runtime version: 0.5.10
  --------------------------------------
  [INFO]: Virtual npu mode is 1_1

  Tools version: 0.6.1.14
  59588c54
  10.8712, 283
  10.6592, 285
  9.3338, 281
  8.8770, 282
  8.1893, 356
  --------------------------------------
  Create handle took 255.04 ms (neu 7.66 ms, axe 0.00 ms, overhead 247.37 ms)
  --------------------------------------
  Repeat 100 times, avg time 4.17 ms, max_time 4.83 ms, min_time 4.14 ms

**检测模型**

对于检测模型, 需要执行对应模型的后处理程序(e.g. ``ax_yolov5s``)才可以实现正确的板上运行.

.. code-block:: bash

  /root/ax-samples # ./ax_yolov5s -m yolov5s.joint -i dog.jpg -r 100
  --------------------------------------
  model file : yolov5s.joint
  image file : dog.jpg
  img_h, img_w : 640 640
  Run-Joint Runtime version: 0.5.10
  --------------------------------------
  [INFO]: Virtual npu mode is 1_1

  Tools version: 0.6.1.14
  59588c54
  run over: output len 3
  --------------------------------------
  Create handle took 490.73 ms (neu 22.06 ms, axe 0.00 ms, overhead 468.66 ms)
  --------------------------------------
  Repeat 100 times, avg time 26.06 ms, max_time 26.83 ms, min_time 26.02 ms
  --------------------------------------
  detection num: 3
  16:  93%, [ 135,  219,  310,  541], dog
  2:  80%, [ 466,   77,  692,  172], car
  1:  61%, [ 169,  116,  566,  419], bicycle

更多关于 ``ax-samples`` 的信息可以访问官方 `github <https://github.com/AXERA-TECH/ax-samples>`_ 获取，在  ``ax-samples`` 对应的 ``ModelZoo`` 中提供了更丰富内容：
  - 预编译的可执行程序（例如 ax_classification, ax_yolov5s）
  - Sample 程序运行依赖的 ``joint`` 模型（例如 mobilenetv2.joint，yolov5s.joint）
  - 测试图片（例如 cat.jpg, dog.jpg）
