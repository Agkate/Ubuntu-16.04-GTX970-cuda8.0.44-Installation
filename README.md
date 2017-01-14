# Ubuntu-16.04-GTX970-cuda8.0.44-Installation
     
首先介绍一下我的电脑配置，我的显卡是NVIDIA GTX970

1. 安装双系统（Ubuntu16.04 + Windows 7）全都是64位的操作系统
          我用U盘制作系统盘安装Ubuntu16.04的时候，遇到无法将启动引导正常安装的问题 
            
            由于Ubuntu14.04安装cuda的时候坑太多，看好几个帖子都这么说的，我还是坚定地想装Ubunt16.04。
            然后参考：从Ubuntu 14.04 LTS版升级到Ubuntu 16.04 LTS。到此，Ubuntu16.04安装成功！

2. 安装NVIDIA显卡驱动
        这里要引用PPA第三方库，因为直接从NVIDIA官方安装，会有显示器黑屏、进入不了tty1界面等一系列问题，没办法，Ubuntu对于NVIDIA显卡驱动的支持不太好
    sudo add-apt-repository ppa:graphics-drivers/ppa    //引入PPA库里的显卡驱动
          
        如果引用成功，则会显示如下图所示：        
   Fresh drivers from upstream, currently shipping Nvidia.

   ## Current Status

   Current official release: `nvidia-370` (370.28)
   Current long-lived branch release: `nvidia-367` (367.57)

   For GeForce 8 and 9 series GPUs use `nvidia-340` (340.98)
   For GeForce 6 and 7 series GPUs use `nvidia-304` (304.132)

   ## What we're working on right now:

   - Normal driver updates
   - Help Wanted: Mesa Updates for Intel/AMD users, ping us if you want to help do this work, we're shorthanded.
           
        接下来安装当前的长期稳定版nvidia-367驱动
    $ sudo service lightdm stop                     #关闭图形桌面，如果不关闭，可能会在安装显卡驱动的时候提示
    $ sudo apt-get install nvidia-367               #X server未关闭的错误，从而导致安装失败
    $ sudo service lightdm start                    #再重新开启图形桌面
    $ sudo reboot                                   #重启一下
    $ nvidia-smi                                    #查看NVIDIA-367显卡驱动安装情况
       如果显卡驱动安装成功，则在执行完nvidia-smi语句后，输出如下：
  Sat Jan 14 10:41:03 2017       
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 367.57                 Driver Version: 367.57                    |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |===============================+======================+======================|
  |   0  GeForce GTX 970     Off  | 0000:01:00.0     Off |                  N/A |
  | 30%   30C    P8    19W / 200W |    121MiB /  4036MiB |      0%      Default |
  +-------------------------------+----------------------+----------------------+
                                                                               
  +-----------------------------------------------------------------------------+
  | Processes:                                                       GPU Memory |
  |  GPU       PID  Type  Process name                               Usage      |
  |=============================================================================|
  |    0      1386    G   /usr/lib/xorg/Xorg                             111MiB |
  |    0      2341    G   compiz                                           8MiB |
  +-----------------------------------------------------------------------------+
    若安装失败，卸载未安装成功的显卡驱动，再重新安装
     $ sudo apt-get remove --purge nvidia-*                   #卸载显卡驱动


3. Cuda安装
         Cuda官方下载地址：https://developer.nvidia.com/cuda-downloads      我用的是 cuda_8.0.44_linux.run 版本       
              
        进入cuda_8.0.44_linux.run 所在目录，执行下面的语句开始安装cuda
     $  sudo sh cuda_8.0.44_linux.run
     可能遇到的选项：
        是否接受许可条款：       accept        
        是否安装NVIDIA driver：no                #因为我们已经安装了NVIDIA显卡驱动
        是否安装cuda toolkit ：   yes
        是否安装cuda samples：yes
        中间会有提示是否确认选择默认路径当作安装路径，按Enter键即可。
     
     若安装失败，且最后错误的提示为：
   Not enough space on parition mounted at /tmp.Need 5091561472 bytes.
   Disk space check has failed. Installation cannot continue.
    即错误提示为/tmp空间不足，可执行下面的操作：
    ================如果执行 $  sudo sh cuda_8.0.44_linux.run 时提示/tmp空间不足，则可以执行下面的操作=====================================
    $ sudo mkdir /opt/tmp                   #在根目录下的opt文件夹中新建tmp文件夹，用作安装文件的临时文件夹
    $ sudo sh cuda_8.0.44_linux.run --tmpdir=/opt/tmp/  
    ================如果执行 $  sudo sh cuda_8.0.44_linux.run 时提示/tmp空间不足，则可以执行上面的操作=====================================
    配置环境变量   
  $ sudo vim  ~/.bashrc                 #打开配置文件，如果没安装vim，可执行 $ sudo apt-get install vim  #安装vim
   按 i 键，在文件末尾插入下面两行，按esc键，输入 :wq ，保存退出。
   export PATH=/usr/local/cuda-8.0/bin:$PATH
   exportLD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
    立即使配置的环境变量生效  
  source ~/.bashrc

 判断cuda是否安装成功
      执行：
       $ nvcc --version

      输出：
    nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2016 NVIDIA Corporation
  Built on Sun_Sep__4_22:14:01_CDT_2016
  Cuda compilation tools, release 8.0, V8.0.44
    则表示安装成功。
   =================若不幸安装失败，执行下面的命令卸载cuda，然后重新安装=================================================
   $ sudo /usr/local/cuda-8.0/bin/uninstall_cuda_8.0.pl                  
    ================若不幸安装失败，执行上面的命令卸载cuda，然后重新安装==================================================
   测试cuda的Samples
   $ cd /usr/local/cuda-8.0/samples/1_Utilities/deviceQuery
   $ make
   $ sudo ./deviceQuery
     输出的最后两行类似这样的信息：
   deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = GeForce GTX 970
   Result = PASS
4.使用Cudnn加速
        我们去官网下载与cuda8.0匹配的cudnn，https://developer.nvidia.com/cudnn ，我下载的是cudnn v5.05 for cuda8.0
        直接将文件解压，拷贝到cuda相应的文件夹下即可
    $ tar xvzf cudnn-8.0-linux-x64-v5.0-ga.tgz
    $ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    $ sudo cp cuda/lib64/*.* /usr/local/cuda/lib64
    $ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

5. 安装编译Caffe
     下载caffe
   $ sudo git clone https://github.com/BVLC/caffe.git
       安装第三方库
   $ sudo apt-get install libatlas-base-dev
   $ sudo apt-get install libprotobuf-dev
   $ sudo apt-get install libleveldb-dev
   $ sudo apt-get install libsnappy-dev
   $ sudo apt-get install libopencv-dev
   $ sudo apt-get install libboost-all-dev
   $ sudo apt-get install libhdf5-serial-dev
   $ sudo apt-get install libgflags-dev
   $ sudo apt-get install libgoogle-glog-dev
   $ sudo apt-get install liblmdb-dev
   $ sudo apt-get install protobuf-compiler
   安装OpenCV
       当前最新版OpenCV是3.2.0版本的
   $ cd caffe
   $ sudo git clone https://github.com/jayrambhia/Install-OpenCV
   $ cd Install-OpenCV/Ubuntu
   $ sudo chmod +x *
   $ sudo ./opencv_latest.sh
     我们可以通过如下命令查看OpenCV安装版本
   $ pkg-config --modversion opencv
     编译caffe
   $ sudo make clean        //每次需要重新编译Caffe的时候，在caffe文件夹下清除掉之前的编译结果

   $ sudo make -j8
   $ sudo make runtest
   $ sudo make pycaffe 
   配置环境
    caffe运行时需要调用cuda的库，我们在/etc/ld.so.conf.d目录下新建一个caffe.conf文件，将所需要用的库的目录写入
   $ sudo vim /etc/ld.so.conf.d/caffe.conf
     添加：  /usr/local/cuda/lib64
     保存并退出      :wq 
     更新配置      
   $ sudo ldconfig
6.测试caffe
     下载mnist数据集
  $ cd ~/caffe                         //切换到caffe目录       注意：执行命令的时候最好在当前的caffe目录下，否则会报错，会找不到XXX文件
  $ sudo sh data/mnist/get_mnist.sh     //获取mnist数据集
  $ sudo sh examples/mnist/create_mnist.sh 
     开始训练
  $ sudo sh examples/mnist/train_lenet.sh
     训练结果  
I0114 13:41:23.117650  4189 solver.cpp:404]     Test net output #0: accuracy = 0.9908
I0114 13:41:23.117681  4189 solver.cpp:404]     Test net output #1: loss = 0.0286537 (* 1 = 0.0286537 loss)
I0114 13:41:23.117684  4189 solver.cpp:322] Optimization Done.
I0114 13:41:23.117687  4189 caffe.cpp:254] Optimization Done.

7. 系统备份与还原
   系统备份
   首先打开终端进入根目录并获取root权限
  $ cd /
  $ sudo su
  # tar cvpzf Ubuntu_backup.tgz --exclude=/Ubuntu_backup.tgz --exclude=/mnt --exclude=/home --exclude=/proc --exclude=/sys --exclude=/lost+found /
    其中 Ubuntu_backup.tgz为备份系统的名字，exclude参数用于设定忽略的文件夹，最后那个/是指示需要备份的目录。备份完后就可以拷贝到其他盘里保存了。
  系统恢复
     在 Ubuntu_backup.tgz 所在文件夹下打开终端获取root权限，将 Ubuntu_backup.tgz拷贝到根目录下
  $ sudo su
  # cp Ubuntu_backup.tgz /
  # cd /
  # tar xvpfz Ubuntu.tgz -C /
   新建备份时忽略的文件夹，如
  # mkdir /proc /lost+found /mnt /sys


[完]


重要参考：Ubuntu16.04 64位 + NVIDIA GEFORECE GTX960 + CUDA-8.0.44   

附加参考：
   Ubuntu sudo update与upgrade的作用及区别
   Linux下 ln 命令详解 
   安装cuda时tmp空间不足问题的解决方法 ：
   Ubuntu16.04+cuda8.0+caffe安装教程
 Caffe 工程的一些编译错误以及解决方案（undefined reference to cv::imread）
    
