import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv

load_dotenv() # Загружаем переменные из .env 
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def setup_logging(log_path):

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    
    file_handler = RotatingFileHandler(log_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
