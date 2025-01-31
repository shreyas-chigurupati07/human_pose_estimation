import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


project_name = 'cnnEstimation'


list_of_files = [
    '.github/workflows/.gitkeep',
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/components/data_ingestion.py',
    f'src/{project_name}/components/prepare_base_model.py',
    f'src/{project_name}/components/training.py',
    f'src/{project_name}/components/evaluation.py',
    f'src/{project_name}/components/prepare_callbacks.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/configuration.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/entity/config_entity.py',
    f'src/{project_name}/constants/__init__.py',
    f'src/{project_name}/pipeline/stage_01_data_ingestion.py',
    f'src/{project_name}/pipeline/stage_02_prepare_base_model.py',
    f'src/{project_name}/pipeline/stage_03_training.py',
    f'src/{project_name}/pipeline/stage_04_evaluation.py',
    f'src/{project_name}/pipeline/predict.py',
    'config/config.yaml',
    'app.py',
    'Dockerfile',
    'dvc.yaml',
    'params.yaml',
    'requirements.txt',
    'setup.py',
    'main.py',
    'research/trails.ipynb',
    'templates/index.html'
]



for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Created directory: {filedir} for the file: {filename}')


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Created empty file: {filename} in the directory: {filedir}') 

    else:
        logging.info(f'File: {filename} already exists in the directory: {filedir}') 
        