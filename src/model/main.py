import sys
import argparse
from loguru import logger
from beartype import beartype
from landmarkcaliper import LandmarkCaliper


@beartype
def run_model(image_file : str) -> list[str]:

    caliper = LandmarkCaliper()

    caliper.measure(image_file)

    return caliper.get_measurement_files()


if __name__ == '__main__':
    
    try:
        
        logger.info(sys.argv)
        
        parser = argparse.ArgumentParser(description='LandmarkCaliper')
        
        parser.add_argument('--image_file', type = str, required = True, help = 'image file path for landmark detection')
        
        args = parser.parse_args()
        
        image_file = args.image_file

        logger.info(f"running landmark model for image: {image_file}")

        run_model(image_file)
            
    except Exception as e:
        print(f'Failed to run landmark model. Error: {e}')
        logger.exception(f'Failed to run landmark model. Error: {e}')
