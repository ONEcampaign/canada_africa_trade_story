import logging


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up the logger.

    Args:
        name (str): The name of the logger.
        level (int): Logging level (default is logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger_ = logging.getLogger(name)
    logger_.setLevel(level)

    if not logger_.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger_.addHandler(console_handler)

    return logger_


logger = setup_logger("research_project")