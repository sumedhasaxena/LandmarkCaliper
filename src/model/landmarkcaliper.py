import os
import sys
import cv2 as cv2
import numpy as np
import pandas as pd
import copy
from typing import NamedTuple
from loguru import logger
from beartype import beartype
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
# from tabulate import tabulate

import matplotlib
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('../lib/')
sys.path.append('../model')
sys.path.append('../data')
sys.path.append('../log')

from utils import *
from modelconfig import ModelConfig
from handlandmarker import HandLandmarkerModel
from distancecalculator import DistanceCalculator
from scalingdistancecalculator import ScalingDistanceCalculator
from worlddistancecalculator import WorldDistanceCalculator


class LandmarkCaliper(object):

    @beartype
    def __init__(self) -> None:

        self.landmark_model = None
        self.measurement_model = None
        self.image_file = None
        self.hand_display_landmarks = None
        self.measurement_files = None

        self.config = ModelConfig()

        self.init_logger()

        self.init_hand_display_landmarks()

        self.init_model()

    @beartype
    def init_logger(self) -> None:

        log_file = self.config.get_log_file()

        if (log_file is None) or len(log_file) == 0:
            return

        log_file = os.path.abspath(log_file)

        if self.config.get_disable_notebook_log():
            logger.remove()

        logger.add(log_file)

    @beartype
    def init_hand_display_landmarks(self) -> None:

        hand_display_landmark_file = self.config.get_hand_display_landmark_file()

        if not check_file_path(hand_display_landmark_file):
            logger.error(f'Failed to read hand display landmarks, file does not exist: {hand_display_landmark_file}')

            return None

        hand_display_landmark_file = os.path.abspath(hand_display_landmark_file)

        logger.info(f'reading hand display landmarks from {hand_display_landmark_file}')

        self.hand_display_landmarks = pd.read_csv(hand_display_landmark_file)

    @beartype
    def init_model(self) -> None:

        logger.info("Initialising landmark detection model...")

        landmark_model_path = self.config.get_landmark_model_path()
        confidence_threshold = self.config.get_landmark_model_confidence_threshold()

        # crate landmark detection model
        self.landmark_model = HandLandmarkerModel(landmark_model_path, confidence_threshold=confidence_threshold)

        logger.info("Initialised landmark detection model.")

        logger.info("Initialising distance calculator...")

        landmark_mapping_file = self.config.get_landmark_mapping_file()
        hand_measurement_file = self.config.get_hand_measurement_file()
        a4_size = self.config.get_a4_measurements()

        model_name = self.config.get_measurement_model_type()

        if model_name == "HAND_GEOM":
            # distance calculator using world landmark co-ordinates
            self.measurement_model = WorldDistanceCalculator(landmark_mapping_file, hand_measurement_file)

        else:
            # distance calculator usnig a4 contours
            self.measurement_model = ScalingDistanceCalculator(landmark_mapping_file, hand_measurement_file,
                                                               a4_size=a4_size)

        logger.info(f"Initialised distance calculator {model_name}.")

    @beartype
    def measure(self, image_file: str) -> None:

        if not check_file_path(image_file):
            return

        self.image_file = image_file

        output_image, landmark_result = self.landmark_model.detect_img_file(image_file)

        if len(landmark_result.handedness) != 0:
            self.measurement_model.init(output_image, landmark_result)
            self.measurement_files = self.save_measurements()
        else:
            logger.error('Cannot initialise measurement model as no results were returned from detection')

    @beartype
    def show_measurement_details(self, show_hand=True) -> None:

        hand_type = self.measurement_model.get_hand_type()
        image_name = get_file_name(self.image_file)
        max_length, max_landmarks = self.measurement_model.get_max_hand_length()
        hand_measurements = self.get_hand_measurements()

        print(f'**** Landmark measurements for {image_name} (cm) ****\n')
        print(
            f'Hand Type: {hand_type}, Length: {round(max_length, 2)} cm b/w {max_landmarks[0]} to {max_landmarks[1]}\n')

        self.show_hand_coordinates_image()

        print(f'\nLandmark Measurements:\n')

        display(hand_measurements)

        if self.measurement_files is not None:
            print(f'\n1. Hand Measurement File: {self.measurement_files[0]} \n')
            print(f'2. Landmark Measurement File: {self.measurement_files[1]} \n')
            print(f'3. Coordinates Image File: {self.measurement_files[2]} \n')
            print(f'4. Contour Image File: {self.measurement_files[3]}\n')
            print(f'4. Landmarks Image File: {self.measurement_files[3]}\n')

    @beartype
    def show_hand_coordinates_image(self, show_pixel_distance=False, fig_size=(10, 15)) -> None:

        image = self.draw_hand_coordinates_and_distance(show_pixel_distance=show_pixel_distance)

        hand_type = self.measurement_model.get_hand_type()
        image_name = get_file_name(self.image_file)

        max_length, max_landmarks = self.measurement_model.get_max_hand_length()

        plt.figure(figsize=fig_size)
        plt.imshow(image)
        plt.title(
            f'Landmark measurements for {image_name} (cm): {hand_type} Hand, Length: {round(max_length, 1)} cm b/w {max_landmarks[0]} to {max_landmarks[1]}')

        plt.show()

        plt.close()

    @beartype
    def show_a4_contours_image(self, show_coordinates=True, font_size=1, fig_size=(10, 15)) -> None:

        image = self.draw_a4_contours(show_coordinates=show_coordinates, font_size=font_size)

        hand_type = self.measurement_model.get_hand_type()
        image_name = get_file_name(self.image_file)

        plt.figure(figsize=fig_size)
        plt.imshow(image)
        plt.title(f'A4 Contours: {hand_type} Hand, {image_name}')

        plt.show()

        plt.close()

    @beartype
    def save_measurements(self, hand_measurements=True, landmark_measurements=True, coordinate_display=True,
                          contour_display=True, landmark_display=True, output_dir=None) -> list[str]:

        hand_type = self.measurement_model.get_hand_type()

        output_file1 = ''
        output_file2 = ''
        output_file3 = ''
        output_file4 = ''
        output_file5 = ''

        if hand_measurements:
            output_file1 = get_hand_measurement_file_name(self.image_file, hand_type, output_dir=output_dir)

            logger.info(f'saving hand measurements to file: {output_file1}')

            self.measurement_model.save_hand_measurements(output_file1)

        if landmark_measurements:
            output_file2 = get_landmark_measurement_file_name(self.image_file, hand_type, output_dir=output_dir)

            logger.info(f'saving landmark measurements to file: {output_file2}')

            self.measurement_model.save_landmark_measurements(output_file2)

        if coordinate_display:
            output_file3 = get_coordinate_display_file_name(self.image_file, hand_type, output_dir=output_dir)

            logger.info(f'saving landmark display image to: {output_file3}')

            # save landmark display
            self.save_hand_coordinate_image(output_file3)

        if contour_display:
            output_file4 = get_contours_display_file_name(self.image_file, hand_type, output_dir=output_dir)

            logger.info(f'saving a4 contours display image to: {output_file4}')

            # save contour display
            self.save_a4_contours_image(output_file4)

        if landmark_display:
            output_file5 = get_landmark_display_file_name(self.image_file, hand_type, output_dir=output_dir)

            logger.info(f'saving landmark display image to: {output_file5}')

            self.save_landmark_display_image(output_file5)

        return [output_file1, output_file2, output_file3, output_file4, output_file5]

    @beartype
    def get_hand_measurements(self) -> pd.DataFrame:

        return self.measurement_model.get_hand_measurements()

    @beartype
    def get_landmark_measurements(self) -> pd.DataFrame:

        return self.measurement_model.get_landmark_measurements()

    @beartype
    def get_measurement_files(self) -> list[str]:

        if self.measurement_files is None or len(self.measurement_files) == 0:
            return []
        else:
            return self.measurement_files

    @beartype
    def save_a4_contours_image(self, file_name) -> None:

        show_coordinates = True
        font_size = 1
        fig_size = (10, 15)

        image = self.draw_a4_contours(show_coordinates=show_coordinates, font_size=font_size)

        hand_type = self.measurement_model.get_hand_type()
        image_name = get_file_name(self.image_file)

        plt.figure(figsize=fig_size)
        plt.imshow(image)
        plt.title(f'A4 Contours: {hand_type} Hand, {image_name}')

        plt.savefig(file_name)

        plt.close()

    @beartype
    def draw_a4_contours(self, show_coordinates=True, font_size=1) -> np.ndarray:

        image = self.measurement_model.get_input_image()
        contours = self.measurement_model.get_a4_contours()

        cv2.drawContours(image, [contours], 0, (0, 0, 255), 1)

        if show_coordinates:

            coordinates = contours.ravel()

            for j in range(0, len(coordinates), 2):
                x = coordinates[j]
                y = coordinates[j + 1]
                text = f'{x}-{y}'

                cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, font_size, (255, 0, 0))

        return image

    @beartype
    def save_hand_coordinate_image(self, file_name: str) -> None:

        show_pixel_distance = False
        fig_size = (10, 15)

        image = self.draw_hand_coordinates_and_distance(show_pixel_distance=show_pixel_distance)

        hand_type = self.measurement_model.get_hand_type()
        image_name = get_file_name(self.image_file)

        max_length, max_landmarks = self.measurement_model.get_max_hand_length()
        img_text = f'Landmark measurements (cm): {hand_type} Hand, {image_name}, Length: {max_length}cm b/w {max_landmarks[0]} to {max_landmarks[1]}'

        # using matplotlib library
        # plt.figure(figsize = fig_size)
        # plt.imshow(image)
        # plt.title(img_text)
        #
        # plt.savefig(file_name)
        #
        # plt.close()

        # using Pillow library
        img = Image.fromarray(image)
        imdraw = ImageDraw.Draw(img)
        imdraw.text((90, 10), img_text, fill=(0, 0, 255))
        img.save(file_name)

    @beartype
    def save_landmark_display_image(self, file_name: str) -> None:
        image = self.landmark_model.draw_landmark_annotations_on_image(self.measurement_model.image, self.measurement_model.model_result)
        self.draw_landmark_Id(image)
        img = Image.fromarray(image)
        img.save(file_name)

    @beartype
    def draw_landmark_Id(self, image):

        landmark_ids = self.measurement_model.get_landmark_ids()
        for landmark_id in landmark_ids:
            x, y, z = self.measurement_model.get_landmark_image_cordinates(landmark_id)
            text = f"{landmark_id}"
            cv2.putText(image, text, (x+2, y+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


    @beartype
    def draw_hand_coordinates_and_distance(self, show_pixel_distance=False) -> np.ndarray:

        image = self.measurement_model.get_input_image()
        landmark_ids = self.measurement_model.get_landmark_ids()

        coordinate_color = (0, 0, 0)
        distance_color = (8, 24, 168)
        coordinate_font_size = 0.35
        distance_font_size = 0.45

        for landmark_id in landmark_ids:
            x, y, z = self.measurement_model.get_landmark_image_cordinates(landmark_id)

            #text = f"{landmark_id}: ({x},{y})"
            text = f"({x},{y})"

            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, coordinate_font_size, coordinate_color, 1, cv2.LINE_AA)

        # show landmarks distances on image
        display_landmarks = self.hand_display_landmarks

        for i in range(len(display_landmarks)):
            landmark1_id = display_landmarks.iloc[i]["HAND_LANDMARK_ID_1"]
            landmark2_id = display_landmarks.iloc[i]["HAND_LANDMARK_ID_2"]

            x1, y1, z1 = self.measurement_model.get_landmark_image_cordinates(landmark1_id)
            x2, y2, z2 = self.measurement_model.get_landmark_image_cordinates(landmark2_id)

            distance = self.measurement_model.get_distance(landmark1_id, landmark2_id)

            pixel_distance = int(self.measurement_model._get_vector_distance(x1, y1, 0, x2, y2, 0))

            ##text = f'{pixel_distance}px, {distance}cm' if show_pixel_distance else f'{landmark1_id}-{landmark2_id}: {round(distance, 1)}cm'
            #text = f'{pixel_distance}px, {distance}cm' if show_pixel_distance else f'{landmark1_id}-{landmark2_id}:{round(distance, 1)}cm'
            text = f'{pixel_distance}px, {distance}cm' if show_pixel_distance else f'{round(distance, 1)}cm'

            x = int(0.5 * (x1 + x2))
            y = int(0.5 * (y1 + y2))

            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, distance_font_size, distance_color, 1, cv2.LINE_AA)

        return image
