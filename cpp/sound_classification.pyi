import numpy as np
from typing import Any, Tuple

class SoundClassificationV2:
    def __init__(self, dev_id: int, bmodel_path: str, threshold: float) -> None:
        pass

    def inference(self, buffer: np.ndarray, index: int) -> Tuple[int, float]:
        """
        buffer: int16 array of audio
        index: classification result

        return: (res, probability) 
        If prob < threshold, res will be classified as background (0). In this case, prob will be useless.
        """
        pass
    
    def get_status(self) -> int:
        """
        return status
        0 for doing init
        1 for init finished
        2 for doing inference
        3 for inderence finished
        """
        pass
    
    def get_threshold(self) -> float:
        """
        get threshold
        """
        pass
    
    def set_threshold(self, threshold: float):
        """
        get threshold
        """
        pass
    
    def set_logger_level(self, level: str):
        """
        set logger level
        """
        pass

    
    
