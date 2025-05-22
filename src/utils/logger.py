import logging
import os
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("LegalAIAssistant")

if __name__ == "__main__":
    logger.info("Logger initialized successfully")
    logger.warning("This is a test warning")
    logger.error("This is a test error")