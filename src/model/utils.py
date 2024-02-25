import os
from loguru import logger
from beartype import beartype
from pathlib import Path

@beartype
def check_file_path(file_path : str) -> bool:

    full_path = os.path.abspath(file_path)

    if os.path.exists(full_path) and os.path.isfile(full_path):

        return True

    logger.error(f'File not found: {os.path.exists(full_path)}, {os.path.isfile(full_path)}, {full_path}')
    
    return False


@beartype
def check_dir_path(dir_path : str) -> bool:

    full_path = os.path.abspath(dir_path)

    if os.path.exists(full_path) and os.path.isdir(full_path):

        return True

    logger.error(f'Directory not found: {os.path.exists(full_path)}, {os.path.isfile(full_path)}, {full_path}')
    
    return False


@beartype
def get_file_name(file_path : str) -> str:

    return Path(file_path).name
    
@beartype
def get_file_stem(file_path : str) -> str:

    return Path(file_path).stem

@beartype
def get_file_extn(file_path : str) -> str:

    return Path(file_path).suffix

@beartype
def get_dir_name(file_path : str) -> str:

    return os.path.dirname(file_path)

@beartype
def check_create_dir(dir_path : str) -> None:
    
    if os.path.isfile(dir_path):
        dir_path = get_dir_name(dir_path)
    
    if not Path(dir_path).exists():

        logger.info(f'creating directory: {dir_path}')

        os.mkdir(dir_path)
       

@beartype
def get_output_dir(input_image_file : str) -> str:

    input_image_file = os.path.abspath(input_image_file)

     #get current input directory
    dir_name = get_dir_name(input_image_file)

    # one level up for patient directory
    patient_dir = get_dir_name(dir_name)

    # create output directory
    output_dir = f'{patient_dir}/output'

    check_create_dir(output_dir)

    return output_dir


@beartype
def get_hand_measurement_file_name(input_image_file : str, hand_type : str, output_dir = None) -> str:

    if output_dir is None:
        output_dir = get_output_dir(input_image_file)
    
    check_create_dir(output_dir)
    
    file_prefix = get_file_stem(input_image_file)

    output_file = f'{output_dir}/{file_prefix}_{hand_type.lower()}_hand_measurements.csv'

    return output_file


@beartype
def get_landmark_measurement_file_name(input_image_file : str, hand_type : str, output_dir = None) -> str:

    if output_dir is None:
        output_dir = get_output_dir(input_image_file)
    
    check_create_dir(output_dir)

    file_prefix = get_file_stem(input_image_file)

    output_file = f'{output_dir}/{file_prefix}_{hand_type.lower()}_landmark_measurements.csv'

    return output_file

@beartype
def get_landmark_display_file_name(input_image_file : str, hand_type : str, output_dir = None) -> str:

    if output_dir is None:
        output_dir = get_output_dir(input_image_file)
    
    check_create_dir(output_dir)

    file_prefix = get_file_stem(input_image_file)

    file_suffix = get_file_extn(input_image_file)

    output_file = f'{output_dir}/{file_prefix}_{hand_type.lower()}_landmarks{file_suffix}'

    return output_file

@beartype
def get_contours_display_file_name(input_image_file : str, hand_type : str, output_dir = None) -> str:

    if output_dir is None:
        output_dir = get_output_dir(input_image_file)
    
    check_create_dir(output_dir)

    file_prefix = get_file_stem(input_image_file)

    file_suffix = get_file_extn(input_image_file)

    output_file = f'{output_dir}/{file_prefix}_{hand_type.lower()}_contours{file_suffix}'

    return output_file


@beartype
def get_measurement_summary_file_name(output_dir : str, hand_type : str) -> str:
    
    output_file = f'{output_dir}/{hand_type.lower()}_hand_measurement_summary.csv'

    return output_file



@beartype
def get_all_hand_measurements_files(output_dir : str) -> list[str]:

    measurement_files = []

    for (root, dirs, files) in os.walk(output_dir):

        for f in files:
            if '_hand_measurements.csv' in f:
                measurement_files.append(f'{output_dir}/{f}')
    
    return measurement_files