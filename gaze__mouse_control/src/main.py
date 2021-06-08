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
import traceback

#@profile
def main(args):
    device=args.device
    video_source=args.video_source
    output_path=args.output_path
    precision=args.precision
    show_output=args.show_output
    LOGGING=0 # Set to 1 to see logs

    model_face = "../intel/face-detection-0202/"+precision+"/face-detection-0202"
    model_landmark = "../intel/facial-landmarks-35-adas-0002/"+precision+"/facial-landmarks-35-adas-0002"
    model_headpose = "../intel/head-pose-estimation-adas-0001/"+precision+"/head-pose-estimation-adas-0001"
    model_gaze = "../intel/gaze-estimation-adas-0002/"+precision+"/gaze-estimation-adas-0002"
     
    start_model_load_time=time.time()
    fd= face_detection_model(model_face, device)
    fd.load_model()
    face_detection_load_time = round(time.time() - start_model_load_time,2)
    if LOGGING: print("{} loaded into device: {}".format(model_face, device))

    start_model_load_time=time.time()
    fl= face_landmark_model(model_landmark, device)
    fl.load_model()
    face_landmark_load_time = round(time.time() - start_model_load_time,2)
    if LOGGING: print("{} loaded into device:{}".format(model_landmark, device))

    start_model_load_time=time.time()
    hp= head_pose_model(model_headpose, device)
    hp.load_model()
    head_pose_load_time = round(time.time() - start_model_load_time,2)
    if LOGGING: print("{} loaded into device:{}".format(model_headpose, device))

    start_model_load_time=time.time()
    gz= gaze_estimation_model(model_gaze, device)
    gz.load_model()
    gaze_estimation_load_time = round(time.time() - start_model_load_time,2)
    if LOGGING: print("{} loaded into device:{}".format(model_gaze, device))

    total_model_load_time = round(face_detection_load_time+face_landmark_load_time+head_pose_load_time+gaze_estimation_load_time, 2)

    try:
        cap=cv2.VideoCapture(video_source)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_source)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
    
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps       = int(cap.get(cv2.CAP_PROP_FPS))
    
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output.mp4'), cv2.VideoWriter_fourcc(*'XVID'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()
    fd_inf_time=0
    fl_inf_time=0
    hp_inf_time=0
    gz_inf_time=0
    mc_update_time=0
    stats=dict()

    stats["precision"]=precision
    stats["device"]=device
    stats["face_detection_load_time"]=face_detection_load_time
    stats["face_landmark_load_time"]=face_landmark_load_time
    stats["head_pose_load_time"]=head_pose_load_time
    stats["gaze_estimation_load_time"]=gaze_estimation_load_time
    stats["total_model_load_time"]=total_model_load_time

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                print("Unable to read video frame!")
                break
            counter+=1
            
            # Flip the frame to see the mirror image
            frame = cv2.flip(frame,1)

            # Face detection 
            fd_time=time.time()
            face_coords = fd.predict(frame)
            face = frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]]
            fd_inf_time += round(time.time()-fd_time,5)*1000
            stats["fd_inf_time"]=fd_inf_time
            if LOGGING: print("Face detection done.")
            
            # Annotations- Face
            if show_output:
                xmin,ymin,xmax,ymax = face_coords
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,0,255),2)

            # Facial lanmark detection 
            fl_time=time.time()
            eye_coords = fl.predict(face)
            fl_inf_time += round(time.time()-fl_time,5)*1000
            stats["fl_inf_time"]=fl_inf_time
            if LOGGING: print("Face landmark detection done.")

            p12x, p12y, p15x, p15y, p14x, p17x, p1y, p2y = eye_coords
            left_eye  = face[p12y:p1y+5, p12x:p14x]
            right_eye = face[p15y:p2y+5, p15x:p17x]

            # Annotations- Left and Right Eye 
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

            # Headpose estimation
            hp_time=time.time()
            ypr = hp.predict(face)
            hp_inf_time += round(time.time()-hp_time,5)*1000
            ypr = np.array(ypr).reshape(1,3)
            stats["hp_inf_time"]=hp_inf_time
            if LOGGING: print("Headpose estimation done.")

            # Gaze estimation            
            gz_time=time.time()
            direction = gz.predict(left_eye, right_eye, ypr)
            gz_inf_time += round(time.time()-gz_time,5)*1000
            x = direction[0][0]
            y = direction[0][1]
            stats["gz_inf_time"]=gz_inf_time
            if LOGGING: print("Gaze estimation done.")           

            # Update Mouse pointer
            mc= MouseController("high","fast")
            mc_time=time.time()
            mc.move(x,y)
            mc_update_time += round(time.time()-mc_time,5)*1000
            stats["mc_update_time"]=mc_update_time

            # FPS calculations
            total_time=time.time()-start_inference_time
            total_inference_time=round(total_time, 1)
            fps=round(counter/total_inference_time,2)
            stats["fps"]=fps

            # Update the stats
            log_stats(stats,output_path,counter)

            # Annotations- Eye direction
            if show_output:
                left_eye_center_x  = int(p12x+(p14x-p12x)/2)
                right_eye_center_x = int(p15x+(p17x-p15x)/2)
                cv2.line(frame, (left_eye_center_x, p12y+10), (int(left_eye_center_x+x*200), int(p12y-y*200)), (0,255,0), 2)
                cv2.line(frame, (right_eye_center_x, p15y+10), (int(right_eye_center_x+x*200), int(p15y-y*200)), (0,255,0), 2)

                # Annotations- Stats
                cv2.putText(frame, "Device: {}, Precision: {}".format(device, precision), (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)
                cv2.putText(frame, "FPS: {}".format(fps), (25, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)
                cv2.putText(frame, "Total model load Time: {}".format(total_model_load_time), (25, 75), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)

                # Write output to file
                out_video.write(frame)
              
                # Show face window
                cv2.imshow("face", frame)
            
            if cv2.waitKey(1) & 0xFF ==27:
                break
            
        # End of While()
        cap.release()
        cv2.destroyAllWindows()

    except Exception:
        print(traceback.format_exc())

def log_stats(stats, output_path, counter):
    now = datetime.datetime.now()
    file_time_stamp= 'stats_{}_{}.txt'.format(now.hour, now.minute)
    with open(os.path.join(output_path,file_time_stamp), 'w') as f:
        f.write("precision:{}".format(stats["precision"])+'\n')
        f.write("device:{}".format(stats["device"])+'\n')
        f.write("---------------\n")
        f.write("Model Load Time (secs) \n")
        f.write("---------------\n")
        f.write("face_detection: "+str(stats["face_detection_load_time"])+'\n')
        f.write("face_landmark: "+str(stats["face_landmark_load_time"])+'\n')
        f.write("head_pose: "+str(stats["head_pose_load_time"])+'\n')
        f.write("gaze_estimation: "+str(stats["gaze_estimation_load_time"])+'\n')
        f.write("total_load_time: "+str(stats["total_model_load_time"])+'\n')
        f.write("---------------\n")
        f.write("Inference Time (msecs) \n")
        f.write("---------------\n")
        f.write("face_detection: "+str(stats["fd_inf_time"]/counter)+'\n')
        f.write("face_landmark: "+str(stats["fl_inf_time"]/counter)+'\n')
        f.write("head_pose: "+str(stats["hp_inf_time"]/counter)+'\n')
        f.write("gaze_estimation: "+str(stats["gz_inf_time"]/counter)+'\n')
        f.write("---------------\n")
        f.write("FPS:{}".format(round(stats["fps"],2))+'\n')
        f.write("---------------"+'\n')
        f.write("mc_update_time (msecs): "+str(round(stats["mc_update_time"]/counter,2)))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device', default='CPU', help='Select the inference device (eg: CPU, GPU, MULTI:CPU,GPU) for all 4 models. Default is CPU.')
    parser.add_argument('--video_source', default='../bin/demo.mp4', help='Path to the video source. For camera source use "CAM". Default is video file at "../bin/demo.mp4".')
    parser.add_argument('--output_path', default='../bin/', help='Set the path to store the output video and stats file. Default is "../bin/".')
    parser.add_argument('--precision', default='FP32', help='Select the Precision (eg: FP32, FP16, FP16-INT8) of the inference models. Default is FP32.')
    parser.add_argument('--show_output', default=0, help="Set '1' to show the annotations on the output. By default annotations are disbaled.")
    args=parser.parse_args()

    if (args.video_source=="CAM"):
        args.video_source=0

    main(args)
