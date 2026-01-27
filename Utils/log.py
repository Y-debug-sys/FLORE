import os

from loguru import logger


def setup_logger(log_file: str = "demo.log"):
    """
    Configure and setup logger with both console and file outputs.
    
    This function initializes the logging system with two sinks:
    1. Console output for INFO level and above messages
    2. File output with rotation for DEBUG level and above messages
    
    The logger configuration includes automatic log rotation, retention policy,
    and compression of old log files to save disk space.
    
    Parameters:
        log_file (str): Path to the log file. Defaults to "demo.log"
        
    Returns:
        logger: Configured logger instance
        
    Example:
        >>> setup_logger("my_app.log")
        >>> logger.info("Application started")
    """

    # Clean up existing log file if present
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except PermissionError:
            logger.remove()
            os.remove(log_file)

    # Remove any previously added log handlers (to avoid duplicate logs)
    logger.remove()

    # Add a console (stdout) log handler for INFO level and above
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )

    # Add a file log handler with rotation for DEBUG level and above
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG"
    )

    return logger


if __name__ == "__main__":
    setup_logger()
    logger.info("Logger setup complete")