# Computer Pointer Controller
This project, uses a gaze detection to control the mouse pointer on the computer. App uses the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. 

## How it works
The app uses InferenceEngine API from Intel's OpenVino ToolKit to build the gaze estimation model which requires three inputs:
* The head pose
* The left eye image
* The right eye image

To get these inputs, three other OpenVino models are used:
* Face Detection
* Head Pose Estimation
* Facial Landmarks Detection.

## The Pipeline
The data frames flow from the input source (video or camera), and then inference of the frames happens through different models and finally the output which represents the direction of the gaze is fed to the mouse controller. The flow of data will look like this:

![image](https://user-images.githubusercontent.com/16221610/121136887-fca6da00-c7ea-11eb-9dcb-ce568a0acc99.png)


## Project Set Up and Installation

### System info:
- MS Windows 10 Enterprise version Build 18363
- Intel Core i7-8665U

### Project directory structure:

1. The project “gaze_mouse_control” directory structure is as follows:
- “bin” directory has the input demo video, output videos, and the stats files
- “intel” directory has all the model files

Note:
Models were downloaded using the model_downloader application on Linux and then copied over to Windows.
Example cmd line on linux: 
python3 /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-0202

- “src” directory has all the source files:
The main.py file contains the main logic of the appicaltion. 
There are separate files for each of the models (with the model’s name) that contains the code for abstracting the class and methods are each model eg: load_model(), predict() etc.
Note: 
- input_feeder.py is unused. 
- main.py.lprof is the output of lineprofiler that can be ignored.

### Installation steps:

1. Download and install Anaconda on Windows 10- https://www.anaconda.com/products/individual.
2. Open Anaconda command prompt and type the following to create and activate the virtual conda environment:
> cd gaze_mouse_control
> conda env create -f openvino_env.yml 
> conda activate openvino_env	

### Running the application:
1. For help with Running the application: 
>cd src
>python main.py --help

2. Run the application in default mode: 
>cd src
>python main.py

In this mode, demo.mp4 will be the input video source. Precision for all models will be FP32. Inference device will be CPU. No output will be displayed. 

Note: 
To end the application hit cltr+c on the keyboard.

Output:

Mouse pointer on screen will follow the gaze of the instructor in demo.mp4.
Output video will be saved in “bin”. Stats will also be saved under “bin” with time stamp.

3. Use camera as input, show output annotations, device=GPU and precision FP16:

>cd src
   >python main.py --video CAM --show_output 1 --device GPU --precision FP16

Note: 

To end the application hit Esc on the keyboard.
   

### Benchmarking and Observations:

1. The bottleneck
The Throughput/FPS of the pipeline suffers badly due to the mouse controller API in pyautogui i.e pyautogui.moveRel()takes about 1 sec. This is the biggest bottle neck in the application!

Running after commenting this API the FPS is 26.

Eg:
>cd src
   >python main.py --video CAM --show_output 1 --device GPU --precision FP16

- precision:FP16
- device:GPU
---------------
- Model Load Time (secs) 
---------------
- face_detection: 21.28
- face_landmark: 67.78
- head_pose: 8.52
- gaze_estimation: 8.18
- total_load_time: 105.76
----------------
- Inference Time (msecs) 
----------------
- face_detection: 11.26759036144579
- face_landmark: 6.679999999999997
- head_pose: 2.264658634538152
- gaze_estimation: 2.1151405622489943
---------------
- FPS:26.49
---------------
- mc_update_time (msecs): 0.0 

2. Comparing CPU v/s GPU inference in FP16

>cd src
>python main.py --video CAM --show_output 1 --device CPU --precision FP16

- precision:FP16
- device:CPU
---------------
- Model Load Time (secs) 
---------------
- face_detection: 0.3
- face_landmark: 0.34
- head_pose: 0.08
- gaze_estimation: 0.09
- total_load_time: 0.81
---------------
- Inference Time (msecs) 
---------------
- face_detection: 25.06295454545453
- face_landmark: 4.213484848484849
- head_pose: 2.538257575757576
- gaze_estimation: 2.677651515151515
---------------
- FPS:19.7
---------------
- mc_update_time (msecs): 0.0

Note:

- Using GPU as the inference device improves the FPS from 19 to 26.

- Using GPU the model load time is very high. However, according to Intel documentation this problem can be solved for subsequent runs by enabling caching of the OpenCL kernels.


3. Edge Cases
Initially in the application I used only p12 and p15 coordinates from face landmark detection results for left eye and right eye corners, and then used fixed lengths to crop a square ROI around the eyes. However, this method failed to scale especially when the face was away from the camera. Hence, switched to using p14 and p17 as well to use four coordinates around both eyes and this produced better results.     

It was also noted that the application does poorly under low light conditions because the face detection fails.




