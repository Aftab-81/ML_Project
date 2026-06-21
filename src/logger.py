import logging
import os
from datetime import datetime
import sys

from src.exception import CustomException

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs")

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



#   if __name__ == "__main__":
#        logging.info("Divide by zero error occurred.")
    

    # means:
"""
    "Run this code only when this file is executed directly."
    "If this file is imported as a module in another file, the code inside this block will not be executed."
"""
