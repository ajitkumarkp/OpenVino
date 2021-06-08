import os
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class face_detection_model:

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold

        try:
            self.core = IECore()
            # print ("self.model_weights:  ", self.model_weights)
            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.input_info))
        self.input_shape=self.model.input_info[self.input_name].input_data.shape
        
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        
    def load_model(self):
        # print ("self.device:  ", self.device)
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
    def predict(self, image):
        self.preprocess_input(image)
        input_dict = {self.input_name:self.input_img}
        self.net.start_async(request_id=0, inputs=input_dict)

        if(self.net.requests[0].wait(-1) == 0):
            result = self.net.requests[0].outputs[self.output_name]

        # infer_request_handle  = self.net.start_async(request_id=0, inputs=input_dict)
        # if(infer_request_handle.wait(-1)==0):
        #     result = infer_request_handle.output_blobs[self.output_name]
        
        coords = self.preprocess_outputs(result)
        coords = self.get_face_coords(coords, image)
        # print("coords:", coords)
        return coords  
        
    def get_face_coords(self, coords, image):
        width  = image.shape[1]
        height = image.shape[0]
        for coord in coords:
            xmin = int(coord[0]*width) 
            ymin = int(coord[1]*height)
            xmax = int(coord[2]*width)
            ymax = int(coord[3]*height)
            box = xmin,ymin,xmax,ymax
        return box
        
    def preprocess_outputs(self, result):
        BBs = result[0][0]
        coords = []
        for BB in BBs:
            if BB[2]> self.threshold:
                x_min, y_min, x_max, y_max = BB[3], BB[4], BB[5], BB[6]
                coord = (x_min, y_min, x_max, y_max)
                coords.append(coord)
        return coords
            
    def preprocess_input(self, image):
        self.input_img = cv2.resize(image, (self.input_shape[3],self.input_shape[2]), interpolation=cv2.INTER_AREA)
        self.input_img = self.input_img.transpose((2,0,1))
        self.input_img = self.input_img.reshape(1, *self.input_img.shape)
#         print ("self.input_img:", self.input_img.shape)
#         return self.input_img
    