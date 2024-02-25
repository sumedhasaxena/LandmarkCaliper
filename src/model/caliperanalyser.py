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
#from tabulate import tabulate

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

        self.left_hand_summary  = pd.DataFrame()
        self.right_hand_summary = pd.DataFrame()

        self.left_hand_distances = pd.DataFrame()
        self.right_hand_distance = pd.DataFrame()

        self.left_hand_file_name  = None
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
    def summarise(self, output_dir : str) -> None:

        if not check_dir_path(output_dir):

            print(f'cannot find output directory: {output_dir}')

            return
        
        self.output_dir = output_dir

        # read all measurements from csv
        lh_measures, rh_measures = self.read_hand_measurements(output_dir)

        # merge all measurements
        lh_merged = self.combine_measurements(lh_measures)
        rh_merged = self.combine_measurements(rh_measures)

        lh_summary, lh_distances = self.summarise_measurements(lh_merged)
        rh_summary, rh_distance  = self.summarise_measurements(rh_merged)

        self.left_hand_summary   = lh_summary
        self.left_hand_distances = lh_distances

        self.right_hand_summary  = rh_summary
        self.right_hand_distance = rh_distance

        # save summary
        lh_file, rh_file = self.save_measurements()

        self.left_hand_file_name    = lh_file
        self.right_hand_file_name   = rh_file


    @beartype
    def show_summary_details(self) -> None:
                                         
        print(f'**** Landmark Caliper Measurements Analysis ****\n')

        print(f'Landmark Caliper Output Directory: {self.output_dir}\n')

        hand_type = "LEFT"

        if len(self.left_hand_summary) > 0: 

            self.show_hand_summary(hand_type, self.left_hand_summary, self.left_hand_distances)

        else:
            print(f'No measurements found for {hand_type} Hand\n')

        hand_type = "RIGHT"

        if len(self.right_hand_summary) > 0: 

            self.show_hand_summary(hand_type, self.right_hand_summary, self.right_hand_distance)

        else:
            print(f'No measurements found for {hand_type} Hand\n')

        print(f'\n1. Left Hand Summary File: {self.left_hand_file_name} \n')
        print(f'2. Right Hand Summary File: {self.right_hand_file_name} \n')


    #@beartype
    def read_hand_measurements(self, output_dir : str) -> any:

        logger.info(f'Reading hand measurements from directory: {output_dir}')

        files = get_all_hand_measurements_files(output_dir)

        logger.info(f'Total hand measurements files: {len(files)}')

        left_measurements  = []
        right_measurements = []

        for f in files:

            hand_measurements = pd.read_csv(f)

            if len(hand_measurements) > 0:

                hand_type = hand_measurements["HAND_TYPE"].iloc[0]

                if hand_type.lower() == "right":

                    right_measurements.append(hand_measurements)

                elif hand_type.lower() == "left":

                    left_measurements.append(hand_measurements)

                else:
                    logger.error(f'invalid hand type from file: {f}')
        
        return (left_measurements, right_measurements)
    

    def combine_measurements(self, hand_measures : list[pd.DataFrame]) -> pd.DataFrame:

        if len(hand_measures) == 0: 
            return pd.DataFrame()
        
        measure1 = hand_measures[0]
        
        measure1 = measure1.rename(columns = {'EUCLIDEAN_DISTANCE':('ED_0')}) 
        
        for i in range(1, len(hand_measures)): 
            
            measure_i = hand_measures[i][["LANDMARK1_ID", "LANDMARK2_ID", "EUCLIDEAN_DISTANCE"]]
            
            measure_i =  measure_i.rename(columns = {'EUCLIDEAN_DISTANCE':(f'ED_{i}')})
            
            measure1 = pd.merge(measure1, measure_i, on= ["LANDMARK1_ID", "LANDMARK2_ID"], how='inner')

        return measure1
    

    def summarise_measurements(self, merged_measurements : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        if len(merged_measurements) == 0:
            return (pd.DataFrame(), pd.DataFrame())

        hand_summary = merged_measurements[['MODEL_TYPE', 'HAND_TYPE', 'LANDMARK_1', 'LANDMARK_2', 'LANDMARK1_ID', 'LANDMARK2_ID']]

        distances = merged_measurements.drop(['MODEL_TYPE', 'HAND_TYPE', 'LANDMARK_1', 'LANDMARK_2', 'LANDMARK1_ID', 'LANDMARK2_ID'], axis = 1)
        
        hand_summary = hand_summary.copy()

        hand_summary["EUCLIDEAN_DISTANCE_MEAN"] = np.round(distances.mean(axis = 1), 3)
        hand_summary["EUCLIDEAN_DISTANCE_STD"]  = np.round(distances.std(axis = 1), 3)

        return (hand_summary, distances)
    
    
   
  

    @beartype
    def show_hand_summary(self, hand_type, hand_summary : pd.DataFrame, hand_distances : pd.DataFrame) -> None:

        sample_size = len(hand_distances.columns)

        print(f'Measurement Summary for {hand_type} Hand for total samples {sample_size}\n')

        print(f'Mean Landmark Measurements:\n')
                            
        display(hand_summary)

        hand_distances.plot(kind="bar", figsize = (15, 7))
        
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

            self.left_hand_summary.to_csv(output_file1, index = False)

        if len(self.right_hand_summary) > 0:

            hand_type = "RIGHT"

            output_file2 = get_measurement_summary_file_name(self.output_dir, hand_type)

            logger.info(f'saving {hand_type} hand summarised measurements to file: {output_file1}')

            self.right_hand_summary.to_csv(output_file2, index = False)
        
        return (output_file1, output_file2)