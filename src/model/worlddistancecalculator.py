import os
import sys
import cv2 as cv2
import numpy as np
import pandas as pd
from typing import NamedTuple
from loguru import logger
from beartype import beartype

import mediapipe as mpipe
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

from distancecalculator import DistanceCalculator

class WorldDistanceCalculator(DistanceCalculator):

    @beartype
    def __init__(self, landmark_mapping_file: str, hand_landmark_file: str, use_z_coordinates = True) -> None:

        super(WorldDistanceCalculator, self).__init__(landmark_mapping_file, hand_landmark_file)

        self.model_name = "HAND_GEOM"

        self.rescaled_world_landmarks = None

        self.use_z_coordinates  = use_z_coordinates

        self.trans_coordindates = None
    
    @beartype
    def calculate(self) -> pd.DataFrame|None:

        # change origin of world coordiante from geometric centre to wrist

        # calculate landmark distances
        self._calculate_landmark_distances()

    @beartype
    def _get_model_name(self) -> str:

        return self.model_name
    
    def _get_euclidean_distance(self, landmark1_id: int|np.int64, landmark2_id: int|np.int64) -> float:
        '''
        Returns euclidean distance between two landmarks in centimeters
        '''

        x1, y1, z1 = self.get_landmark_world_cordinates(landmark1_id)

        x2, y2, z2 = self.get_landmark_world_cordinates(landmark2_id)

        if not self.use_z_coordinates:
            z1 = 0
            z2 = 0

        #distance = self._get_vector_distance(x1, y1, z1, x2, y2, z2)
        
        distance1 = self._get_vector_distance(x1, y1, z1, 0, 0, 0)
        distance2 = self._get_vector_distance(x2, y2, z2, 0, 0, 0)

        distance = distance1 + distance2

        # round upto millimeter
        return round(distance, 3)
    

    def _transform_world_coordinates(self, mult = 1) -> None:
        
        # model_points you can get from results.multi_hand_world_landmarks
        # image points from results.multi_hand_landmarks
        # to get the camera matrix and the distortion ideally you would calibrate your camera, 
        # but you can use approximate values like so (just substitute your frame width and height)
        # pseudo camera internals
        
        #frame_height, frame_width, channels = (720, 1280, 3)

        frame_height, frame_width, channels = self.image.shape


        model_points = np.float32([[-l.x, -l.y, -l.z] for l in self.model_result.hand_world_landmarks[0]])

        image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in self.model_result.hand_landmarks[0]])
        
        focal_length = frame_width * mult
        center       = (frame_width/2, frame_height/2)
        
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], 
                                  dtype = "double")
        
        distortion = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points,
                                                                    camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
        
        # you can set various flags, but the SQPNP seemed to perform the best
        # Apply the found transformation to the model coordinates like so:
        
        transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
        
        transformation[0:3, 3] = translation_vector.squeeze()
        
        # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate
        
        # transform model coordinates into homogeneous coordinates
        
        model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)
        
        # apply the transformation
        world_points = model_points_hom.dot(np.linalg.inv(transformation).T)

        #and then in world_points  you have the 'real' 3D coordinates of the hand
        self.trans_coordindates = world_points
