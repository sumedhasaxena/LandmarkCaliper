import json
import os
from beartype import beartype
from loguru import logger

class ModelConfig:
    
    @beartype
    def __init__(self, file = ""):
        
        if len(file) <= 0:
              file = __file__
        
        with open(os.path.join(os.path.dirname(file), 'modelconfig.json')) as f:
            self.config = json.load(f)
    
    @beartype
    def get_config(self) -> dict:
        
        return self.config['app_config']
    
    @beartype
    def _get_value(self, key_name : str) -> str|int|float|bool:
        
        return self.get_config()[key_name]
    
    @beartype
    def get_a4_measurements(self) -> tuple[float, float]:
         
        width   = self._get_value("a4_width_cms")
        heiight = self._get_value("a4_height_cms")

        return (width, heiight)
    
    @beartype
    def get_landmark_model_path(self) -> str:

        return self._get_value("landmark_model_path")
        
    @beartype
    def get_landmark_model_confidence_threshold(self) -> float:

        return self._get_value("landmark_model_confidence_threshold")
    
    @beartype
    def get_landmark_mapping_file(self) -> str:

        return self._get_value("landmark_mapping_file")
    
    @beartype
    def get_hand_measurement_file(self) -> str:

        return self._get_value("hand_measurement_file")
    
    @beartype
    def get_hand_display_landmark_file(self) -> str:

        return self._get_value("hand_display_landmark_file")
    
    @beartype
    def get_measurement_model_type(self) -> str:
        
        return self._get_value("measurement_model_type")
    
    @beartype
    def get_log_file(self) -> str:

        return self._get_value("app_log_file")
    

    @beartype
    def get_disable_notebook_log(self) -> bool:

        return self._get_value("disable_notebook_log")
    
    
if __name__ == "__main__":
    
    config = ModelConfig()
    
    print(config.get_config())