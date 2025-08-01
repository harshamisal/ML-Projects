import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(os.path.dirname(logs_path), exist_ok = True)
LOG_FILE_PATH = os.path.join(os.getcwd(), 'logs', LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = '[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level = logging.INFO
)


# if __name__ =  = '__main__':
#     logging.info('This is a info message')
#     logging.warning('This is a warning message')
#     logging.error('This is a error message')
#     logging.critical('This is a critical message')
#     logging.debug('This is a debug message')