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

from model.caliperanalyser import CaliperAnalyser

class NotebookAnalyser(object):

    def __init__(self) -> None:

        # file chooser

        self.fc = FileChooser('../data/')

        #fc.filter_pattern = ['*.jpg', '*.png']

        self.fc.title = '<b>Select Landmark Caliper Output Directory</b>'


    def choose_output_dir(self) -> None:

        display(self.fc)

    @beartype
    def run_analysis(self) -> None:
        
        try:

            # caliper            
            output_dir = self.fc.selected_path

            if len(output_dir) > 0:

                analyser = CaliperAnalyser()
                                                
                analyser.summarise(output_dir)

                analyser.show_summary_details()
                
            else:
                print('Select Landmark Caliper output directory')
                        
        except Exception as e:
            print(f'Failed to run Landmark Caliper Analyser. Error: {e}')
            logger.exception(f'Failed to run Landmark Caliper Analyser. Error: {e}')

