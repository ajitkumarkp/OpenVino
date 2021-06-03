import os
import sys
import logging as log
from openvino.inference_engine import IECore
import cv2

'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class face_landmark_model:

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

        self.input_name=next(iter(self.model.input_info))
        self.input_shape=self.model.input_info[self.input_name].input_data.shape

        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

        # print("Input name and shape:", self.input_name,"and", self.input_shape)
        # print("Output name and shape:", self.output_name,"and", self.output_shape)

        
    def load_model(self):
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
    def predict(self, image):
        self.preprocess_input(image)
        input_dict = {self.input_name:self.input_img}
        self.net.start_async(request_id=0, inputs=input_dict)

        if(self.net.requests[0].wait(-1) == 0):
            result = self.net.requests[0].outputs[self.output_name]
        
        coords = self.preprocess_outputs(result, image)
        return coords, image 
        
    def preprocess_outputs(self, result, image):
        width  = image.shape[1]
        height = image.shape[0]

        p12x = int(result[0][24]*width)
        p12y = int(result[0][25]*height)
        
        p15x = int(result[0][30]*width)
        p15y = int(result[0][31]*height)

        # cv2.rectangle(image, (p12x,p12y), (p12x+60,p12y+60), (255,0,0),2)
        # cv2.rectangle(image, (p15x,p15y), (p15x+60,p15y+60), (255,0,0),2)
        
        # cv2.circle(image,(p12x, p12y),5,(255,255,0),thickness=-1,lineType=cv2.FILLED)
        # cv2.circle(image,(p15x, p15y),5,(255,255,0),thickness=-1,lineType=cv2.FILLED)
        
        return p12x, p12y, p15x, p15y
            
    def preprocess_input(self, image):
        self.input_img = cv2.resize(image, (self.input_shape[3],self.input_shape[2]), interpolation=cv2.INTER_AREA)
        self.input_img = self.input_img.transpose((2,0,1))
        self.input_img = self.input_img.reshape(1, *self.input_img.shape)
    