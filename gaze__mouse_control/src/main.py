import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
from face_detection import face_detection_model
from face_landmark_detection import face_landmark_model
from head_pose_estimation import head_pose_model
from gaze_estimation import gaze_estimation_model
import matplotlib.pyplot as plt 
from mouse_controller import MouseController

def main(args):
    model_face = "../intel/face-detection-0202/FP16/face-detection-0202"
    model_landmark = "../intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002"
    model_headpose = "../intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001"
    model_gaze = "../intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002"
    
    device=args.device
    video_file=args.video
    threshold=args.threshold
    output_path=args.output_path

    # start_model_load_time=time.time()
    
    fd= face_detection_model(model_face, device)
    fd.load_model()
    
    fl= face_landmark_model(model_landmark, device)
    fl.load_model()
    
    hp= head_pose_model(model_headpose, device)
    hp.load_model()

    gz= gaze_estimation_model(model_gaze, device)
    gz.load_model()

    # total_model_load_time = time.time() - start_model_load_time

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps       = int(cap.get(cv2.CAP_PROP_FPS))
    # out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    # start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            
            face_coords, image = fd.predict(frame)         
            face = frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]

            eye_coords, image = fl.predict(face)
            left_eye = image[eye_coords[1]:eye_coords[1]+60, eye_coords[0]:eye_coords[0]+60]
            right_eye = image[eye_coords[3]:eye_coords[3]+60, eye_coords[2]:eye_coords[2]+60]

            ypr = hp.predict(face)
            ypr = np.array(ypr).reshape(1,3)
            
            direction = gz.predict(left_eye, right_eye, ypr)
            x = direction[0][0]
            y = direction[0][1]
            # print (x, y)
            mc= MouseController("high","fast")
            mc.move(x,y)
            # print ("yaw, pitch, roll :", ypr.shape)

            # plt.imshow(face[...,::-1]); plt.show()
            cv2.imshow("face", frame)
            if cv2.waitKey(25) & 0xFF ==27:
                break
            # out_video.write(image)

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    # parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=0)
    parser.add_argument('--output_path', default='../')
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()

    main(args)
