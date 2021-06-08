import os
import sys
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

        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape

        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        
    def load_model(self):
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        
    def predict(self, left_eye, right_eye, ypr):
        self.preprocess_input(left_eye, right_eye)
        
        input_dict = {'left_eye_image':self.left_eye, 'right_eye_image':self.right_eye, 'head_pose_angles': ypr}
        self.net.start_async(request_id=0, inputs=input_dict)

        if(self.net.requests[0].wait(-1) == 0):
            result = self.net.requests[0].outputs[self.output_name]
        
        return result 
        
    def preprocess_input(self, left_eye, right_eye):
        image = cv2.resize(left_eye, (self.input_shape[3],self.input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2,0,1))
        self.left_eye = image.reshape(1,3,self.input_shape[3],self.input_shape[2])
        
        image = cv2.resize(right_eye, (self.input_shape[3],self.input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2,0,1))
        self.right_eye = image.reshape(1,3,self.input_shape[3],self.input_shape[2])
    