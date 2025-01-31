from cnnEstimation.config.configuration import ConfigurationManager
from cnnEstimation.components.data_ingestion import DataIngestion
from cnnEstimation import logger



STAGE_NAME = 'Data Ingestion'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.extract_zip_file()
        


if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>> {STAGE_NAME} - Started <<<<<<<<<<')
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>> {STAGE_NAME} - Completed <<<<<<<<<<')

    except Exception as e:
        logger.error(f'Error in {STAGE_NAME}: {str(e)}')
        raise e