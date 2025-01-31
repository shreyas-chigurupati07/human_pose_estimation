import os
import urllib.request as request
import zipfile
from cnnEstimation import logger
from cnnEstimation.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def extract_zip_file(self):
        """
        Extracts the zip file to the data directory (unzip_dir).
        Function returns None.
        """
        unzip_path = self.config.unzip_dir
        zip_file_path = self.config.local_data_file

        if not os.path.exists(zip_file_path):
            raise FileNotFoundError(f'Zip file not found at {zip_file_path}')

        os.makedirs(unzip_path, exist_ok=True)

        logger.info(f'Extracting {zip_file_path} to {unzip_path}')

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f'Extraction complete: {zip_file_path} to {unzip_path}') 