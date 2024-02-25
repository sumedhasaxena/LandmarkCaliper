import sys
import argparse
from loguru import logger
from beartype import beartype
from IPython.display import display
from ipyfilechooser import FileChooser

import sys
import matplotlib

sys.path.append('..')
sys.path.append('../model')
sys.path.append('../data')

from model.landmarkcaliper import LandmarkCaliper

class NotebookCaliper(object):

    def __init__(self) -> None:

        # file chooser

        self.fc = FileChooser('../data/')

        #fc.filter_pattern = ['*.jpg', '*.png']

        self.fc.title = '<b>Select Patient Hand Image</b>'


    def choose_file(self) -> None:

        display(self.fc)

    @beartype
    def run_model(self) -> None:
        
        try:

            # caliper            
            image_file = self.fc.selected

            if len(image_file) > 0:

                caliper = LandmarkCaliper()
                                                
                caliper.measure(image_file)
                
                caliper.show_measurement_details()
            else:
                print('Select input image first')
                        
        except Exception as e:
            print(f'Failed to run landmark model. Error: {e}')
            logger.exception(f'Failed to run landmark model. Error: {e}')

