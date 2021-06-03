import os
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class gaze_estimation_model:

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_names = self.model.input_info.keys()
        # print("head_pose_angles shape:", self.input_name[0].shape) 
        # self.input_names = self.model.input_info['left_eye_image']

        print("Input :", self.input_names)
        
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        print("Output name and shape:", self.output_name,"and", self.output_shape)
        
    def load_model(self):
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
    def predict(self, left_eye, right_eye, ypr):
        self.preprocess_input(left_eye, right_eye)
        # print ("Left:", self.left_eye.shape)
        
        input_dict = {'left_eye_image':self.left_eye, 'right_eye_image':self.right_eye, 'head_pose_angles': ypr}
        self.net.start_async(request_id=0, inputs=input_dict)

        if(self.net.requests[0].wait(-1) == 0):
            result = self.net.requests[0].outputs[self.output_name]
        
        # print ("result:",result)
        # coords = self.preprocess_outputs(result)
        # coords, image = self.draw_outputs(coords, image)
        return result 
        
    def draw_outputs(self, coords, image):
        width  = image.shape[1]
        height = image.shape[0]
        final_coords=[]
        for coord in coords:
            xmin =  int(coord[0]*width) 
            ymin = int(coord[1]*height)
            xmax = int(coord[2]*width)
            ymax = int(coord[3]*height)
            image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0),2)
            box = (xmin,ymin,xmax,ymax)
            final_coords.append(box)
        return final_coords, image
        
    def preprocess_outputs(self, result):
        BBs = result[0][0]
        coords = []
        for BB in BBs:
            if BB[2]> self.threshold:
                x_min, y_min, x_max, y_max = BB[3], BB[4], BB[5], BB[6]
                coord = (x_min, y_min, x_max, y_max)
                coords.append(coord)
        return coords
            
    def preprocess_input(self, left_eye, right_eye):
        image = cv2.resize(left_eye, (60,60), interpolation=cv2.INTER_AREA)
        image = image.transpose((2,0,1))
        self.left_eye = image.reshape(1,3,60,60)
        
        image = cv2.resize(right_eye, (60,60), interpolation=cv2.INTER_AREA)
        image = image.transpose((2,0,1))
        self.right_eye = image.reshape(1,3,60,60)
#         print ("self.input_img:", self.input_img.shape)
#         return self.input_img
    