import os
import sys
import logging


logging_str = '[%(asctime)s - %(levelname)s - %(module)s - %(message)s]'


log_dir = 'artifacts/logs'
log_filepath = os.path.join(log_dir, 'running_logs.log')

try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f'Error creating log directory: {e}')
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger(__name__)