from xai_framework.types import Writer
import pandas as pd

class LocalWriter(Writer):
    """
    A class for writing data to a local file.

    Args:
        output_dir (str): The directory where the output file will be saved.

    Attributes:
        output_dir (str): The directory where the output file will be saved.

    """
    def __init__(self, output_dir="./output") -> None:
        super().__init__(name="LocalWriter")
        self.output_dir = output_dir
    
    def write(self, df: pd.DataFrame):
        """
        Writes the given DataFrame to a CSV file in the specified output directory.

        Args:
            df (pd.DataFrame): The DataFrame to be written.

        """
        df.to_csv(self.output_dir + "output.csv", index=False)

