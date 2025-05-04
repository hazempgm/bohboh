# src/utils.py
import logging
import sys
from pathlib import Path
from typing import Union, Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
):
    """
    Configures the root logger for the project.

    Allows logging to console and optionally to a file. This function should
    ideally be called once at the start of the main script or application entry point.

    Args:
        level (int): The minimum logging level (e.g., logging.DEBUG, logging.INFO).
        log_file (Optional[Union[str, Path]]): Path to the log file. If None, only logs to console.
        log_format (str): The format string for log messages.
        date_format (str): The format string for the timestamp in log messages.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level) # Set the minimum level for the root logger

    # Remove existing handlers to avoid duplicate logs if called multiple times
    # (though ideally it's called only once)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level) # Console handler uses the specified level
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        try:
            log_file_path = Path(log_file)
            # Ensure the directory exists
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
            file_handler.setLevel(level) # File handler also uses the specified level
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging configured. Level: {logging.getLevelName(level)}. Outputting to console and file: {log_file_path}")
        except Exception as e:
            logging.error(f"Failed to configure file logging to {log_file}: {e}", exc_info=True)
            # Continue with console logging only
    else:
         logging.info(f"Logging configured. Level: {logging.getLevelName(level)}. Outputting to console only.")

# You could add other utility functions here as needed, e.g.:
# def check_path_exists(path: Union[str, Path]): ...
# def format_time(seconds: float): ...


# Example Usage (can be run directly for testing)
if __name__ == '__main__':
    print("--- Testing default logging (should only go to console) ---")
    # Note: Because logging is configured globally, the basicConfig in other
    # modules might interfere if they were imported before this setup runs.
    # In a real application, call setup_logging ONCE near the start.
    setup_logging(level=logging.DEBUG) # Set level to DEBUG for testing

    # Get a logger instance for this specific module test
    test_logger = logging.getLogger(__name__)

    test_logger.debug("This is a debug message (default setup).")
    test_logger.info("This is an info message (default setup).")
    test_logger.warning("This is a warning message (default setup).")

    print("\n--- Testing logging to file ---")
    log_filename = "test_utils.log"
    try:
        # Configure logging to include a file
        setup_logging(level=logging.INFO, log_file=log_filename)

        test_logger.debug("This debug message should NOT appear (level is INFO).")
        test_logger.info("This info message should appear in console and file.")
        test_logger.warning("This warning message should appear in console and file.")
        test_logger.error("This is an error message.")

        print(f"\nCheck the file '{log_filename}' for log output.")

        # Clean up the test log file afterwards
        # import os
        # try:
        #     os.remove(log_filename)
        #     print(f"Removed test log file: {log_filename}")
        # except OSError:
        #     pass # Ignore if file doesn't exist

    except Exception as e:
        print(f"An error occurred during file logging test: {e}")

    print("\nUtility tests finished.")

