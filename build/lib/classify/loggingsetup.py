import logging
from dotenv import load_dotenv

logging.basicConfig(filename="Main_errors.log", level=logging.INFO)
x = "all good"
logging.info(x)

load_dotenv()
