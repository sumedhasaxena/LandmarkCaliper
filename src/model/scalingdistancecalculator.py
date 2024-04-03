import os
import sys
import cv2 as cv2
import numpy as np
import pandas as pd
from loguru import logger
from beartype import beartype
import copy

import mediapipe as mpipe
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

from distancecalculator import DistanceCalculator

class ScalingDistanceCalculator(DistanceCalculator):

    @beartype
    def __init__(self, landmark_mapping_file: str, hand_landmark_file: str, a4_size = (21, 29.7),
                 gauss_ksize = (5, 5), filter_threshold = 190, filter_maxval = 255, use_z_coordinates = False) -> None:
 
        super(ScalingDistanceCalculator, self).__init__(landmark_mapping_file, hand_landmark_file)

        self.model_name     = "A4_CONTOURS"
        self.A4_WIDTH_CMS   = a4_size[0]
        self.A4_HEIGHT_CMS  = a4_size[1]
        self.scaling_factor = 1.0

        # contour related parameters
        self.gauss_ksize        = gauss_ksize
        self.filter_threshold   = filter_threshold
        self.filter_maxval      = filter_maxval
        self.use_z_coordinates  = use_z_coordinates

        self.contour_image = None
        self.a4_contours   = None

    @beartype
    def calculate(self) -> pd.DataFrame|None:

        # calculate scaling factor using A4 contours
        self.calculate_scaling_factor()

        # calculate landmark distances
        self._calculate_landmark_distances()


    @beartype
    def _get_model_name(self) -> str:

        return self.model_name
    
    def _get_euclidean_distance(self, landmark1_id: int|np.int64, landmark2_id: int|np.int64) -> float:
        '''
        Returns euclidean distance between two landmarks in centimeters
        '''

        x1, y1, z1 = self.get_landmark_image_cordinates(landmark1_id)

        x2, y2, z2 = self.get_landmark_image_cordinates(landmark2_id)

        if not self.use_z_coordinates:
            z1 = 0
            z2 = 0

        distance = self._get_vector_distance(x1, y1, z1, x2, y2, z2)

        distance *= self.scaling_factor


        # round upto millimeter
        return round(distance, 3)
    

    @beartype
    def calculate_scaling_factor(self) -> None:
        
        logger.info(f'Calculating pixel scaling factor...')

        # find contours of A4 paper in the image 
        self.find_A4_contours(self.image)

        # find scaling factor using a4 contours
        self.find_optimal_scaling_factor()


    @beartype
    def find_A4_contours(self, image : np.ndarray) -> None:

        # convert image to gray scale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_gray = cv2.GaussianBlur(image_gray, self.gauss_ksize, 0)

        _, threshold_image = cv2.threshold(image_gray, self.filter_threshold, self.filter_maxval, cv2.THRESH_BINARY)

        threshold_image = cv2.erode(threshold_image, None, iterations = 3)

        threshold_image = cv2.dilate(threshold_image, None, iterations = 3)

        # Detecting contours in filtered image

        self.contour_image = threshold_image

        contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = max(contours, key = cv2.contourArea)

        epsilon = 0.01 * cv2.arcLength(largest_contour, True)

        self.a4_contours = cv2.approxPolyDP(largest_contour, epsilon, True)

        logger.info(f'detected a4 countours: {len(self.a4_contours)}')


    @beartype
    def find_optimal_scaling_factor(self) -> None:
        '''
        Detect max X and Y coordinates of A4 contours and use them for real-world size mappnig
        '''
                        
        x_last      = None
        y_last      = None
        x_max       = 0
        y_max       = 0 
        x_max_dist  = 0
        y_max_dist  = 0

        contours = self.a4_contours.ravel()
    
        for i in range(0, len(contours), 2):

            x1 = contours[i]
            y1 = contours[i + 1]
            
            if x_last is not None:

                if x_max < abs(x_last - x1):
                    # update current max x coordinate and corresponding distance between these two contours
                    x_max       = abs(x_last - x1)
                    x_max_dist  = self._get_vector_distance(x_last, y_last, 0, x1, y1, 0)
                    
                if y_max < abs(y_last - y1):
                    # update current max y coordinate and corresponding distance between these two contours
                    y_max       = abs(y_last - y1)
                    y_max_dist  = self._get_vector_distance(x_last, y_last, 0, x1, y1, 0)
    
            x_last = x1
            y_last = y1
                    
        scale_x = round(self.A4_WIDTH_CMS / max(x_max, x_max_dist), 6)
        
        scale_y = round(self.A4_HEIGHT_CMS / max(y_max, y_max_dist), 6)

        self.scaling_factor = min(scale_x, scale_y)

        logger.info(f'Calculated scaling factor: {self.scaling_factor} on Max(X, Y): ({x_max}, {y_max}), Diistance(X, Y): ({x_max_dist}, {y_max_dist})')

    @beartype
    def find_optimal_scaling_factor_OLD(self) -> None:
        '''
        Detect max X and Y coordinates of A4 contours and use them for real-world size mappnig
        '''

        contours = self.a4_contours
                        
        x1, y1 = contours[0][0][0], contours[0][0][1]

        x2, y2 = contours[1][0][0], contours[1][0][1]

        pixel_distance = self._get_vector_distance(x1, y1, 0,  x2, y2, 0)

        width  = abs(x2 - x1)
        height = abs(y2 - y1)

        benchmark_cms = self.A4_WIDTH_CMS if width > height else self.A4_HEIGHT_CMS

        # centimeters per pixel
        self.scaling_factor = round(benchmark_cms / pixel_distance, 6)

        logger.info(f'Calculated scaling factor: {self.scaling_factor} on A4 benchamrk {benchmark_cms} cms, (Wiidth, Height): ({width}, {height})')

    
    def get_a4_contours(self) -> np.ndarray:

        if self.a4_contours is None:
            return None
        
        return copy.deepcopy(self.a4_contours)