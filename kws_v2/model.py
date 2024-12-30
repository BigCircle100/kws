import numpy as np
from typing import Any, Tuple
import sophon.sail as sail
import SILK2.Tools.logger as Logger

class SoundClassificationV2:
    def __init__(self, dev_id: int, bmodel_path: str, threshold: float, logger_level: str="INFO") -> None:
        self.status = 0
        self.logger = Logger.Log("service_kws/model", logger_level)
        try: 
            self.net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
        except Exception as e:
            self.logger.error(f"{Logger.file_lineno()} {str(e)}",exc_info=True)
        self.threshold = threshold
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_name = self.net.get_output_names(self.graph_name)[0]
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)
        self.status = 1
        


    def inference(self, buffer: np.ndarray) -> Tuple[int, float]:
        """
        buffer: int16 array of audio, sample rate 8000

        return: (res, probability) 
        If prob < threshold, res will be classified as background (0). In this case, prob will be useless.
        """

        self.status = 2

        # normalize by dividing the mean of 500 top value in the data 
        data = buffer.astype(np.float32)/32767.0

        if len(data) != 16000:
            self.logger.error(f"{Logger.file_lineno()} input length should be 16000",exc_info=True)

        max_rate = 0.2
        top_n = 500

        top_values = np.partition(np.abs(data), -top_n)[-top_n:]  
        mean_top_values = np.mean(top_values)
        data = data/mean_top_values*max_rate
        data = data[np.newaxis, :]

        input_data = {self.input_name: data}
        output_data = self.net.process(self.graph_name, input_data)
        output = output_data[self.output_name]

        # softmax
        e_x = np.exp(output - np.max(output))
        e_x = e_x / np.sum(e_x)

        # if prob < threshold, return class 0 (background)
        if (np.max(e_x) < self.threshold):
            return 0, e_x[0]     

        self.status = 3 

        return int(np.argmax(e_x)), float(np.max(e_x))

    def get_status(self) -> int:
        return self.status
    
    def get_threshold(self) -> float:
        return self.threshold
    
    def set_threshold(self, threshold: float):
        self.threshold = threshold
    

    
    
