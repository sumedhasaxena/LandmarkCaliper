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


class CaliperAnalyser(object):

    @beartype
    def __init__(self) -> None:

        self.config = ModelConfig()

        self.left_hand_summary = pd.DataFrame()
        self.right_hand_summary = pd.DataFrame()

        self.left_hand_distances = pd.DataFrame()
        self.right_hand_distance = pd.DataFrame()

        self.left_hand_error_summary = pd.DataFrame()
        self.right_hand_error_summary = pd.DataFrame()

        self.left_hand_file_name = None
        self.right_hand_file_name = None

        self.output_dir = None

        self.init_logger()

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
    def summarise(self, output_dir: str) -> None:

        if not check_dir_path(output_dir):
            print(f'cannot find output directory: {output_dir}')

            return

        self.output_dir = output_dir

        # read all measurements from csv
        lh_measures, rh_measures = self.read_hand_measurements(output_dir)

        # merge all measurements
        lh_merged = self.combine_measurements(lh_measures)
        rh_merged = self.combine_measurements(rh_measures)

        lh_error_summary = self.summarise_measurement_errors(lh_measures)
        rh_error_summary = self.summarise_measurement_errors(rh_measures)

        lh_summary, lh_distances = self.summarise_measurements(lh_merged, lh_error_summary)
        rh_summary, rh_distance = self.summarise_measurements(rh_merged, rh_error_summary)

        self.left_hand_summary = lh_summary
        self.left_hand_distances = lh_distances
        self.left_hand_error_summary = lh_error_summary

        self.right_hand_summary = rh_summary
        self.right_hand_distance = rh_distance
        self.right_hand_error_summary = rh_error_summary

        # save summary
        lh_file, rh_file = self.save_measurements()

        self.left_hand_file_name = lh_file
        self.right_hand_file_name = rh_file

    @beartype
    def show_summary_details(self) -> None:

        print(f'**** Landmark Caliper Measurements Analysis ****\n')

        print(f'Landmark Caliper Output Directory: {self.output_dir}\n')

        hand_type = "LEFT"

        if len(self.left_hand_summary) > 0:

            self.show_hand_summary(hand_type, self.left_hand_summary, self.left_hand_distances,
                                   self.left_hand_error_summary)

        else:
            print(f'No measurements found for {hand_type} Hand\n')

        hand_type = "RIGHT"

        if len(self.right_hand_summary) > 0:

            self.show_hand_summary(hand_type, self.right_hand_summary, self.right_hand_distance,
                                   self.right_hand_error_summary)

        else:
            print(f'No measurements found for {hand_type} Hand\n')

        print(f'\n1. Left Hand Summary File: {self.left_hand_file_name} \n')
        print(f'2. Right Hand Summary File: {self.right_hand_file_name} \n')

    # @beartype
    def read_hand_measurements(self, output_dir: str) -> any:

        logger.info(f'Reading hand measurements from directory: {output_dir}')

        files = get_all_hand_measurements_files(output_dir)

        logger.info(f'Total hand measurements files: {len(files)}')

        left_measurements = []
        right_measurements = []

        for f in files:

            hand_measurements = pd.read_csv(f)
            hand_measurements.Name = get_image_name_from_hand_measurement_file(f)

            if len(hand_measurements) > 0:

                hand_type = hand_measurements["HAND_TYPE"].iloc[0]

                if hand_type.lower() == "right":

                    right_measurements.append(hand_measurements)

                elif hand_type.lower() == "left":

                    left_measurements.append(hand_measurements)

                else:
                    logger.error(f'invalid hand type from file: {f}')

        return (left_measurements, right_measurements)

    @beartype
    def combine_measurements(self, hand_measures: list[pd.DataFrame]) -> pd.DataFrame:

        if len(hand_measures) == 0:
            return pd.DataFrame()

        measure1 = hand_measures[0]

        measure1 = measure1.rename(columns={'EUCLIDEAN_DISTANCE': (f'ED_{measure1.Name}')})

        for i in range(1, len(hand_measures)):
            measure_i = hand_measures[i][["LANDMARK1_ID", "LANDMARK2_ID", "EUCLIDEAN_DISTANCE"]]

            measure_i = measure_i.rename(columns={'EUCLIDEAN_DISTANCE': (f'ED_{hand_measures[i].Name}')})

            measure1 = pd.merge(measure1, measure_i, on=["LANDMARK1_ID", "LANDMARK2_ID"], how='inner')

        return measure1

    def summarise_measurement_errors(self, hand_measures: list[pd.DataFrame]) -> pd.DataFrame:

        if len(hand_measures) == 0:
            return pd.DataFrame()

        error_summary = None

        for i in range(len(hand_measures)):

            measure = hand_measures[i]

            total_abs_error = np.round(measure['ERROR'].abs().mean(), 3)

            total_pct_error = np.round(measure['ERROR_PCT'].abs().mean(), 3)

            error_measure = pd.DataFrame(
                {'IMAGE': [f'ED_{measure.Name}'], 'AVG_ERROR': [total_abs_error], 'AVG_PCT_ERROR': [total_pct_error]})

            if error_summary is None:
                error_summary = error_measure
            else:
                error_summary = pd.concat([error_summary, error_measure])

        error_summary = error_summary.reset_index(drop=True)

        error_summary = error_summary.sort_values(by=['AVG_ERROR', 'AVG_PCT_ERROR'])

        return error_summary

    def summarise_measurements(self, merged_measurements: pd.DataFrame, error_summary: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame]:

        if len(merged_measurements) == 0:
            return (pd.DataFrame(), pd.DataFrame())

        # select all columns except distance measurements
        hand_summary = merged_measurements[
            ['MODEL_TYPE', 'HAND_TYPE', 'LANDMARK_1', 'LANDMARK_2', 'LANDMARK1_ID', 'LANDMARK2_ID',
             'PHYSICAL_DISTANCE']]

        # drop all columns excpet measurements across images
        distances = merged_measurements.drop(
            ['MODEL_TYPE', 'HAND_TYPE', 'LANDMARK_1', 'LANDMARK_2', 'LANDMARK1_ID', 'LANDMARK2_ID', 'PHYSICAL_DISTANCE',
             'ERROR', 'ERROR_PCT'], axis=1)

        hand_summary = hand_summary.copy()

        # take average of distances and their standard deviation
        hand_summary["EUCLIDEAN_DISTANCE_MEAN"] = np.round(distances.mean(axis=1), 3)
        hand_summary["EUCLIDEAN_DISTANCE_STD"] = np.round(distances.std(axis=1), 3)

        # calculate mean distance measurement errors compared to physical distance
        physical_distance = hand_summary["PHYSICAL_DISTANCE"]
        mean_distance = hand_summary["EUCLIDEAN_DISTANCE_MEAN"]

        # check if physical distances are available to calculate error and % errors
        if np.sum(physical_distance) > 0:
            error = np.round(mean_distance - physical_distance, 2)
            error_pct = np.round(100 * error.divide(physical_distance.to_numpy()), 2)
            error = np.where(physical_distance <= 0, 0, error)
            error_pct = np.where(physical_distance <= 0, 0, error_pct)
        else:
            error = np.zeros(len(physical_distance))
            error_pct = np.zeros(len(physical_distance))

        # add error columns to hand summary
        hand_summary["ERROR"] = error
        hand_summary["ERROR_PCT"] = error_pct

        summary_columns = ['MODEL_TYPE', 'HAND_TYPE', 'LANDMARK_1', 'LANDMARK_2', 'LANDMARK1_ID', 'LANDMARK2_ID',
                           'EUCLIDEAN_DISTANCE_MEAN', 'EUCLIDEAN_DISTANCE_STD', 'PHYSICAL_DISTANCE', 'ERROR',
                           'ERROR_PCT']

        hand_summary = hand_summary[summary_columns]

        # put distances in their sorted order of error
        distances = distances.loc[:, error_summary['IMAGE']]

        # add physical distance measurements distance data frame
        distances["MEAN_DISTANCE"] = hand_summary["EUCLIDEAN_DISTANCE_MEAN"]
        distances["PHYSICAL_DISTANCE"] = hand_summary["PHYSICAL_DISTANCE"]

        return (hand_summary, distances)

    @beartype
    def show_hand_summary(self, hand_type, hand_summary: pd.DataFrame, hand_distances: pd.DataFrame,
                          error_summary: pd.DataFrame) -> None:

        sample_size = max(0, len(hand_distances.columns) - 2)

        print(f'Measurement Summary for {hand_type} Hand for total samples {sample_size}\n')

        print(f'Mean Landmark Measurements:\n')

        display(hand_summary)

        hand_distances.plot(kind="bar", figsize=(15, 7))
        hand_distances.plot(figsize=(15, 7))

        print(f'\nLandmark Error Summary:')
        display(error_summary)

        plt.title(f'{hand_type} Landmark Measurements Across {sample_size} images')

        plt.show()

    @beartype
    def save_measurements(self) -> tuple[str, str]:

        output_file1 = ''
        output_file2 = ''

        if len(self.left_hand_summary) > 0:
            hand_type = "LEFT"

            output_file1 = get_measurement_summary_file_name(self.output_dir, hand_type)

            logger.info(f'saving {hand_type} hand summarised measurements to file: {output_file1}')

            self.left_hand_summary.to_csv(output_file1, index=False)

        if len(self.right_hand_summary) > 0:
            hand_type = "RIGHT"

            output_file2 = get_measurement_summary_file_name(self.output_dir, hand_type)

            logger.info(f'saving {hand_type} hand summarised measurements to file: {output_file1}')

            self.right_hand_summary.to_csv(output_file2, index=False)

        return (output_file1, output_file2)