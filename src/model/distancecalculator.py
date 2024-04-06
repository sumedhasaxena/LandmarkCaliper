import os
import sys
import numpy as np
import pandas as pd
import copy
from loguru import logger
from beartype import beartype
from abc import ABC
from typing import Optional

from utils import *
import mediapipe as mpipe
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

sys.path.append("../data/")


class DistanceCalculator(ABC):

    @beartype
    def __init__(self, landmark_mapping_file: str, hand_landmark_file: str) -> None:

        self.landmark_mapping = self.read_landmark_mappings(landmark_mapping_file)
        self.hand_landmarks = self.read_hand_landmarks(hand_landmark_file)

        self.image = None
        self.model_result = None
        self.image_height = 0
        self.image_width = 0
        self.image_path = None

        self.landmark_distance = None
        self.hand_measurements = None
        self.physical_measurements = None
        self.max_hand_length = None
        self.max_hand_length_landmarks = None

    @beartype
    def init(self, image: mpipe.Image, landmark_result: HandLandmarkerResult, image_file: str) -> None:

        self.image = image.numpy_view()
        self.model_result = landmark_result
        self.image_path = image_file

        self.image_height = self.image.shape[0]
        self.image_width = self.image.shape[1]

        # read physical measurements if available
        self.physical_measurements = self.read_hand_physical_measurements()

        # calculate landmark distances
        self.calculate()

    @beartype
    def calculate(self) -> pd.DataFrame | None:
        pass

    @beartype
    def _get_euclidean_distance(self, landmark1_id: int | np.int64, landmark2_id: int | np.int64) -> float:
        '''
        Returns euclidean distance between two landmarks in centimeters
        '''
        pass

    @beartype
    def _get_model_name(self) -> str:
        pass

    def _calculate_landmark_distances(self) -> None:

        hand_type = self.get_hand_type()
        model_name = self._get_model_name()

        logger.info(f"calculating landmark distances using model {model_name} for {hand_type} hand...")

        landmark_ids = np.sort(np.unique(self.landmark_mapping["ID"]))

        landmark_distances = None
        max_hand_length = 0
        max_hand_length_landmarks = None

        for landmark1_id in landmark_ids:

            landmark1_name = self.get_landmark_name(landmark1_id)

            for landmark2_id in landmark_ids:

                if landmark2_id <= landmark1_id:
                    continue

                euclidean_distance = self._get_euclidean_distance(landmark1_id, landmark2_id)

                landmark2_name = self.get_landmark_name(landmark2_id)

                physical_distance = self.get_physical_measurement(landmark1_id, landmark2_id)

                if physical_distance > 0.:
                    error = round(euclidean_distance - physical_distance, 2)
                    error_pct = round(100 * error / physical_distance, 2)
                else:
                    error = 0.
                    error_pct = 0.

                landmark_dist = pd.DataFrame({"MODEL_TYPE": [model_name],
                                              "HAND_TYPE": [hand_type],
                                              "LANDMARK_1": [landmark1_name],
                                              "LANDMARK_2": [landmark2_name],
                                              "LANDMARK1_ID": [landmark1_id],
                                              "LANDMARK2_ID": [landmark2_id],
                                              "EUCLIDEAN_DISTANCE": [euclidean_distance],
                                              "PHYSICAL_DISTANCE": [physical_distance],
                                              "ERROR": [error],
                                              "ERROR_PCT": [error_pct]})

                if landmark_distances is None:
                    landmark_distances = landmark_dist
                else:
                    landmark_distances = pd.concat((landmark_distances, landmark_dist), axis=0)

                # record max length of the hand
                if euclidean_distance > max_hand_length:
                    max_hand_length = euclidean_distance
                    max_hand_length_landmarks = (landmark1_name, landmark2_name)

        self.landmark_distance = landmark_distances
        self.max_hand_length = max_hand_length
        self.max_hand_length_landmarks = max_hand_length_landmarks

        self._update_hand_measurements()

        logger.info(
            f'calculated landmark distances and updated hand measurements. total: {len(self.landmark_distance)}')

    def _update_hand_measurements(self) -> None:

        logger.info(f"updating hand measurements...")

        measurements = None

        for i in range(len(self.hand_landmarks)):

            landmark1_id = self.hand_landmarks.iloc[i]["HAND_LANDMARK_ID_1"]
            landmark2_id = self.hand_landmarks.iloc[i]["HAND_LANDMARK_ID_2"]

            distance = self.get_distance_data(landmark1_id, landmark2_id)

            if measurements is None:
                measurements = pd.DataFrame(distance)
            else:
                measurements = pd.concat((measurements, pd.DataFrame(distance)), axis=0)

        measurements = measurements.reset_index(drop=True)

        self.hand_measurements = measurements

        logger.info(f'updated hand measurements. total: {len(self.hand_measurements)}')

    # @beartype
    def _get_vector_distance(self, x1: float | int, y1: float | int, z1: float | int,
                             x2: float | int, y2: float | int, z2: float | int) -> float:

        return round(((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5, 6)

    @beartype
    def get_landmark_name(self, landmark_id: int | np.int64) -> str:

        landmark_map = self.landmark_mapping[self.landmark_mapping["ID"] == landmark_id]

        return landmark_map["HAND_LANDMARK"].iloc[0] if len(landmark_map) > 0 else "UNKNOWN_LANDMARK"

    @beartype
    def get_physical_measurement(self, landmark1_id: int | np.int64, landmark2_id: int | np.int64) -> float:

        if (self.physical_measurements is None) or len(self.physical_measurements) == 0:
            return 0.

        measurement = self.physical_measurements[(self.physical_measurements["LANDMARK1_ID"] == landmark1_id) &
                                                 (self.physical_measurements["LANDMARK2_ID"] == landmark2_id)]

        if len(measurement) == 0:
            measurement = self.physical_measurements[(self.physical_measurements["LANDMARK1_ID"] == landmark2_id) &
                                                     (self.physical_measurements["LANDMARK2_ID"] == landmark1_id)]

        return float(measurement['DISTANCE'].iloc[0]) if len(measurement) > 0 else 0.

    @beartype
    def get_distance_data(self, landmark1_id: int | np.int64, landmark2_id: int | np.int64) -> pd.DataFrame:

        distance = self.landmark_distance.loc[(self.landmark_distance["LANDMARK1_ID"] == landmark1_id) &
                                              (self.landmark_distance["LANDMARK2_ID"] == landmark2_id),]

        return distance

    @beartype
    def get_distance(self, landmark1_id: int | np.int64, landmark2_id: int | np.int64) -> float:

        distance = self.get_distance_data(landmark1_id, landmark2_id)

        return distance["EUCLIDEAN_DISTANCE"].iloc[0] if len(distance) > 0 else 0.

        return distance

    @beartype
    def get_hand_type(self) -> str:

        handedness = self.model_result.handedness[0][0]
        return handedness.display_name

        # "LEFT" if handedness.category_name == "Right" else "RIGHT"

    @beartype
    def get_landmark_image_cordinates(self, landmark_id: int | np.int64) -> tuple[int, int, int]:

        coordinate = self.model_result.hand_landmarks[0][landmark_id]

        x = int(coordinate.x * self.image_width)
        y = int(coordinate.y * self.image_height)
        z = int(coordinate.z * self.image_width)

        return (x, y, z)

    @beartype
    def get_landmark_world_cordinates(self, landmark_id: int | np.int64) -> tuple[float, float, float]:
        '''
        Returns landmark's world coordinates in centimeters
        '''

        coordinate = self.model_result.hand_world_landmarks[0][landmark_id]

        # world coordinates are in meter - convert them into centimeter

        x = coordinate.x * 100
        y = coordinate.y * 100
        z = coordinate.z * 100

        return (x, y, z)

    @beartype
    def get_landmark_ids(self) -> np.ndarray:

        landmark_ids = np.sort(np.unique(self.landmark_mapping["ID"]))

        return landmark_ids

    @beartype
    def get_max_hand_length(self) -> tuple:

        return (self.max_hand_length, self.max_hand_length_landmarks)

    @beartype
    def read_landmark_mappings(self, mapping_file: str) -> pd.DataFrame | None:

        if not check_file_path(mapping_file):
            logger.error(f'Failed to read landmark mapping, file does not exist: {mapping_file}')

            return None

        mapping_file = os.path.abspath(mapping_file)

        logger.info(f'reading landmark mappings from {mapping_file}')

        landmark_mapping = pd.read_csv(mapping_file)

        return landmark_mapping

    @beartype
    def read_hand_landmarks(self, hand_landmark_file: str) -> pd.DataFrame | None:

        if not check_file_path(hand_landmark_file):
            logger.error(f'Failed to read hand landmarks, file does not exist: {hand_landmark_file}')

            return None

        hand_landmark_file = os.path.abspath(hand_landmark_file)

        logger.info(f'reading hand landmarks from {hand_landmark_file}')

        hand_landmarks = pd.read_csv(hand_landmark_file)

        return hand_landmarks

    @beartype
    def read_hand_physical_measurements(self) -> Optional[pd.DataFrame]:

        try:

            hand_type = self.get_hand_type()

            measurement_file = get_hand_physical_measurement_file_name(self.image_path, hand_type)

            if not check_file_path(measurement_file):
                logger.error(f'skipped reading physical measurements, file does not exist: {measurement_file}')

                return None

            measurement_file = os.path.abspath(measurement_file)

            logger.info(f'reading hand physical measurements from {measurement_file}')

            measurements = pd.read_csv(measurement_file)

            if len(measurements.columns) == 5:
                measurements.columns = ['LANDMARK_1', 'LANDMARK_2', 'LANDMARK1_ID', 'LANDMARK2_ID', 'DISTANCE']

            return measurements

        except Exception as ex:
            logger.error(f'skipped reading physical measurements, error in reading file: {ex}')

    @beartype
    def save_landmark_measurements(self, file_path: str) -> None:

        if (self.landmark_distance is None) or (len(self.landmark_distance) == 0):
            logger.error(f'Landmark measurements are not calculated, ignored saving to file')

            return

        logger.info(f'saving landmark measurements to {file_path}')

        self.landmark_distance.to_csv(file_path, index=False)

    @beartype
    def save_hand_measurements(self, file_path: str) -> None:

        if (self.hand_measurements is None) or (len(self.hand_measurements) == 0):
            logger.error(f'Hand measurements are not calculated, ignored saving to file')

            return

        logger.info(f'saving hand measurements to {file_path}')

        self.hand_measurements.to_csv(file_path, index=False)

    @beartype
    def get_input_image(self) -> np.ndarray:
        '''
        Returns deep copy of current input image
        '''

        return copy.deepcopy(self.image)

    def get_hand_measurements(self) -> pd.DataFrame:
        '''
        Returns copy of current hand measurements
        '''

        return copy.deepcopy(self.hand_measurements)

    def get_landmark_measurements(self) -> pd.DataFrame:
        '''
        Returns copy of current landmark measurements
        '''

        return copy.deepcopy(self.landmark_distance)


