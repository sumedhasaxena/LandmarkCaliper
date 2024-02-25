import sys
import os
import numpy as np
from loguru import logger
from beartype import beartype

import mediapipe as mpipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

sys.path.append("../lib/")

class HandLandmarkerModel(object):

    @beartype
    def __init__(self, model_path : str, confidence_threshold = 0.7) -> None:
        
        self.model_path             = model_path
        self.confidence_threshold   = confidence_threshold

        self.model = self.init_model(model_path)

    @beartype
    def init_model(self, model_path : str) -> HandLandmarker:

        full_path = os.path.abspath(model_path)

        if not self.check_file_path(full_path):

            logger.error(f'Failed to init model, cannot find the base model from path: {full_path}')

            return None

        logger.info(f'Initialising model from {full_path}, confidence_threshold: {self.confidence_threshold}')

        base_options = python.BaseOptions(model_asset_path = model_path)

        options = vision.HandLandmarkerOptions(base_options = base_options, 
                                               num_hands = 2, 
                                               min_hand_detection_confidence = self.confidence_threshold, 
                                               min_hand_presence_confidence = self.confidence_threshold)

        model = vision.HandLandmarker.create_from_options(options)

        return model
    
    #@beartype
    def detect_img_array(self, image_array: np.array) -> tuple[mpipe.Image|None, HandLandmarkerResult|None]:

        if (image_array is None) or (len(image_array) == 0): 

            logger.error(f'Invalid input array for image')

            return None
        
        logger.info(f'Detecting landmarks on image')

        image = mpipe.Image(image_format = mpipe.ImageFormat.SRGB, data = image_array)

        result = self.run_model(image)
        
        return result
    
    @beartype
    def detect_img_file(self, image_path : str) -> tuple[mpipe.Image|None, HandLandmarkerResult|None]:

        full_path = os.path.abspath(image_path)

        if not self.check_file_path(full_path): 

            logger.error(f'Failed to find image from path: {full_path}')

            return None
        
        logger.info(f'Detecting landmarks on image: {full_path}')

        image = mpipe.Image.create_from_file(full_path)

        result = self.run_model(image)

        return (image, result) 
    
    @beartype
    def run_model(self, image : mpipe.Image) -> HandLandmarkerResult:
        '''
        Returns model result for landmark detection.
        1. Result.handedness: represents whether the detected hands are left or right hands.

        2. result.hand_landmarks: there are 21 hand landmarks, each composed of x, y and z coordinates. 
            The x and y coordinates are normalized to [0.0, 1.0] by the image width and height, respectively. 
            The z coordinate represents the landmark depth, with the depth at the wrist being the origin. 
            The smaller the value, the closer the landmark is to the camera. 
            The magnitude of z uses roughly the same scale as x.

        3. result.hand_world_landmarks: the 21 hand landmarks are also presented in world coordinates. 
            Each landmark is composed of x, y, and z, representing real-world 3D coordinates in meters with 
            the origin at the hands geometric center.
        '''

        result = self.model.detect(image)

        hand = result.handedness[0][0]

        logger.info(f'Detected landmarks. Category: {hand.category_name}, Confidence: {round(hand.score * 100, 2)}')

        return result
    
    @beartype
    def check_file_path(self, file_path : str) -> bool:

        full_path = os.path.abspath(file_path)

        if os.path.exists(full_path) and os.path.isfile(full_path):

            return True

        logger.error(f'{os.path.exists(full_path)}, {os.path.isfile(full_path)}, {full_path}')
        
        return False