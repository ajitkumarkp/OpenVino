
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 30
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    # Note - CPU extensions are moved to plugin since OpenVINO release 2020.1. 
    # The extensions are loaded automatically while     
    # loading the CPU plugin, hence 'add_extension' need not be used.

 
    #      parser.add_argument("-l", "--cpu_extension", required=False, type=str,
    #                         default=None,
    #                         help="MKLDNN (CPU)-targeted custom layers."
    #                              "Absolute path to a shared library with the"
    #                              "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(IPADDRESS, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_BB(frame, result, width, height, prob_threshold):
    BBs = result[0][0]
    out_frame = frame
    count_ppl_per_frame = 0
    for BB in BBs:
        if BB[2]> prob_threshold:
            count_ppl_per_frame+=1
            x_min, y_min, x_max, y_max = BB[3], BB[4], BB[5], BB[6]
            top_left =  (int(x_min*width), int(y_min*height))
            bot_right = (int(x_max*width), int(y_max*height))
            out_frame = cv2.rectangle(frame, top_left, bot_right, (255,0,0),2)
    return out_frame, count_ppl_per_frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, CPU_EXTENSION)
    net_input_shape = infer_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    cam_frame_no = 0
    timer_running = False
    start = 0
    duration = 0
    frames_wo_person = 0
    total_count = 0
    new_person = False 

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        cam_frame_no += 1

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.async_inference(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.extract_output()
            
            ### TODO: Extract any desired stats from the results ###
            out_frame, count_ppl_per_frame = draw_BB(frame, result, width, height, prob_threshold)
            
        if count_ppl_per_frame: # person detected
            if frames_wo_person >=3: # new detection after >=3 frames w/o any detections => its a new person
                new_person = True
                total_count+=1
                frames_wo_person = 0
                if timer_running:
                    current = time.time()
                    duration = int(current-start)
                else:            
                    start = time.time()
                    timer_running = True        
            else: # Means it is the same person
                new_person = False 
                frames_wo_person = 0
                if timer_running:
                    current = time.time()
                    duration = int(current-start)
                
        else: # person NOT detected
            frames_wo_person += 1
            if frames_wo_person >=3:
                duration = 0
                start = 0
                timer_running = False
                
        ## TODO: Calculate and send relevant information on ###
        ## current_count, total_count and duration to the MQTT server ###
        current_count = count_ppl_per_frame
            
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        client.publish("person", json.dumps({"count":current_count, "total":total_count}))
        client.publish("person/duration", json.dumps({"duration":duration}))

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###

        # Break if escape key pressed
        if key_pressed == 27:
            break
        # End of while loop

    # Release the capture and destroy any OpenCV windows
    cap.release()

    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()

    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
#     client = connect_mqtt()
    client =0
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()

    
    
