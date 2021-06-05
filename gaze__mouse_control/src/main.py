import numpy as np
import time
import datetime
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
    device=args.device
    video_file=args.video
    threshold=args.threshold
    output_path=args.output_path
    precision=args.precision
    show_output=args.show_output

    model_face = "../intel/face-detection-0202/"+precision+"/face-detection-0202"
    model_landmark = "../intel/facial-landmarks-35-adas-0002/"+precision+"/facial-landmarks-35-adas-0002"
    model_headpose = "../intel/head-pose-estimation-adas-0001/"+precision+"/head-pose-estimation-adas-0001"
    model_gaze = "../intel/gaze-estimation-adas-0002/"+precision+"/gaze-estimation-adas-0002"
     
    start_model_load_time=time.time()
    fd= face_detection_model(model_face, device)
    fd.load_model()
    face_detection_load_time = round(time.time() - start_model_load_time,2)

    start_model_load_time=time.time()
    fl= face_landmark_model(model_landmark, device)
    fl.load_model()
    face_landmark_load_time = round(time.time() - start_model_load_time,2)

    start_model_load_time=time.time()
    hp= head_pose_model(model_headpose, device)
    hp.load_model()
    head_pose_load_time = round(time.time() - start_model_load_time,2)

    start_model_load_time=time.time()
    gz= gaze_estimation_model(model_gaze, device)
    gz.load_model()
    gaze_estimation_load_time = round(time.time() - start_model_load_time,2)

    total_model_load_time = round(face_detection_load_time+face_landmark_load_time+head_pose_load_time+gaze_estimation_load_time, 2)


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
    
    counter=0
    start_inference_time=time.time()
    fd_inf_time=0
    fl_inf_time=0
    hp_inf_time=0
    gz_inf_time=0

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1

            # Flip the frame to see the mirror image
            frame = cv2.flip(frame,1)

            fd_time=time.time()
            face_coords = fd.predict(frame)
            face = frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]
            fd_inf_time += round(time.time()-fd_time,3)*1000

            if show_output:
                xmin,ymin,xmax,ymax = face_coords
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,0,255),2)

            fl_time=time.time()
            eye_coords = fl.predict(face)
            fl_inf_time += round(time.time()-fl_time,3)*1000

            p12x, p12y, p15x, p15y, p14x, p17x, p1y, p2y = eye_coords
            left_eye  = face[p12y:p1y+5, p12x:p14x]
            right_eye = face[p15y:p2y+5, p15x:p17x]

            if show_output:
                p12x += face_coords[0] 
                p15x += face_coords[0] 
                
                p12y += face_coords[1] 
                p15y += face_coords[1] 
                
                p14x += face_coords[0]
                p17x += face_coords[0]

                p1y  += face_coords[1] 
                p2y  += face_coords[1] 

                cv2.rectangle(frame, (p12x,p12y), (p14x,p1y+5), (255,0,0),2)
                cv2.rectangle(frame, (p15x,p15y), (p17x,p2y+5), (255,0,0),2)

            hp_time=time.time()
            ypr = hp.predict(face)
            hp_inf_time += round(time.time()-hp_time,3)*1000

            ypr = np.array(ypr).reshape(1,3)
            
            gz_time=time.time()
            direction = gz.predict(left_eye, right_eye, ypr)
            gz_inf_time += round(time.time()-gz_time,3)*1000

            x = direction[0][0]
            y = direction[0][1]

            mc= MouseController("high","fast")
            mc.move(x,y)

            if show_output:
               left_eye_center_x = int(p12x+(p14x-p12x)/2)
               right_eye_center_x = int(p15x+(p17x-p15x)/2)

               cv2.line(frame, (left_eye_center_x, p12y+10), (int(left_eye_center_x+x*200), int(p12y-y*200)), (0,255,0), 2)
               cv2.line(frame, (right_eye_center_x, p15y+10), (int(right_eye_center_x+x*200), int(p15y-y*200)), (0,255,0), 2)
        
            total_time=time.time()-start_inference_time
            total_inference_time=round(total_time, 1)
            fps=round(counter/total_inference_time,2)
            
            y=25
            cv2.putText(frame, "Device: {}, Precision: {}".format(device, precision), (25, y), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, "FPS: {}".format(fps), (25, y+25), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, "Total model load Time: {}".format(total_model_load_time), (25, y+50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)

            cv2.imshow("face", frame)
            # plt.imshow(frame[...,::-1]); plt.show()

            if cv2.waitKey(25) & 0xFF ==27:
                break
            # out_video.write(image)
        
        # print ("model_location:", model_face)
        # print ("precision:",precision)
        # print ("device:",device)
        # print("---------------")
        # print("Model Load Time (secs)")
        # print("---------------")
        # print("face_detection: "+str(face_detection_load_time))
        # print("face_landmark: "+str(face_landmark_load_time))
        # print("head_pose: "+str(head_pose_load_time))
        # print("gaze_estimation: "+str(gaze_estimation_load_time))
        # print("total_load_time: "+str(total_model_load_time))
        # print("---------------")
        # print("Avg inference Time (msecs)")
        # print("---------------")
        # print("face_detection: "+str(round(fd_inf_time/counter,2)))
        # print("face_landmark: "+str(round(fl_inf_time/counter,2)))
        # print("head_pose: "+str(round(hp_inf_time/counter,2)))
        # print("gaze_estimation: "+str(round(gz_inf_time/counter,2)))
        # print("---------------")
        # print("FPS:", round(fps,2))

        now = datetime.datetime.now()
        with open(os.path.join(output_path, 'stats_{}_{}:{}.txt'.format(device, now.hour, now.minute)), 'w') as f:
            # f.write(str(fps)+'\n') \n")
            f.write("precision:{}".format(precision)+'\n')
            f.write("device:{}".format(device)+'\n')
            f.write("---------------\n")
            f.write("Model Load Time (secs) \n")
            f.write("---------------\n")
            f.write("face_detection: "+str(face_detection_load_time)+'\n')
            f.write("face_landmark: "+str(face_landmark_load_time)+'\n')
            f.write("head_pose: "+str(head_pose_load_time)+'\n')
            f.write("gaze_estimation: "+str(gaze_estimation_load_time)+'\n')
            f.write("total_load_time: "+str(total_model_load_time)+'\n')
            f.write("---------------\n")
            f.write("Inference Time (msecs) \n")
            f.write("---------------\n")
            f.write("face_detection: "+str(fd_inf_time/counter)+'\n')
            f.write("face_landmark: "+str(fl_inf_time/counter)+'\n')
            f.write("head_pose: "+str(hp_inf_time/counter)+'\n')
            f.write("gaze_estimation: "+str(gz_inf_time/counter)+'\n')
            f.write("FPS:{}".format(round(fps,2))+'\n')
            f.write("---------------")

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default='../bin/demo.mp4')
    parser.add_argument('--output_path', default='../')
    parser.add_argument('--threshold', default=0.60)
    parser.add_argument('--precision', default='FP32')
    parser.add_argument('--show_output', default=0)
    
    args=parser.parse_args()

    if (args.video=="CAM"):
        args.video=0

    main(args)
