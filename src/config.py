# src/config.py
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Configure logging for this module specifically
# Note: This might be overridden by setup_logging in utils if called later
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path (Union[str, Path]): The path to the YAML configuration file.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the configuration settings
                                  if successful, None otherwise.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) # Use safe_load for security
        logger.info(f"Successfully loaded configuration from: {config_path}")
        return config_data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}", exc_info=True)
        return None

# --- Example Usage ---
# Typically, you wouldn't run this directly. Instead, your main script
# would import load_config and call it.
if __name__ == '__main__':
    logger.info("Testing config module...")

    # Define the path to the example config file relative to the project root
    # Adjust this path if your execution context is different
    default_config_file = Path(__file__).parent.parent / "config" / "default_params.yaml"
    logger.info(f"Attempting to load config file: {default_config_file}")


    # Create a dummy config file if it doesn't exist for testing purposes
    if not default_config_file.exists():
        logger.warning(f"Example config file {default_config_file} not found. Creating a dummy one for testing.")
        default_config_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_config_content = """
data_base_dir: "./data/"
dataset_name: "dummy_dataset"
reconstruction_algorithm: "fbp"
fbp_filter: "ramp"
results_base_dir: "./results/"
log_level: "DEBUG"
log_file: null
"""
        try:
            with open(default_config_file, 'w') as f:
                f.write(dummy_config_content)
            logger.info(f"Created dummy config file: {default_config_file}")
        except Exception as e:
            logger.error(f"Failed to create dummy config file: {e}")
            # Exit if we can't create the dummy file for the test
            sys.exit(1)


    # --- Test Loading ---
    config = load_config(default_config_file)

    if config:
        logger.info("Configuration loaded successfully.")
        # Accessing settings like a dictionary
        print("\n--- Loaded Configuration Settings ---")
        for key, value in config.items():
            print(f"{key}: {value} (Type: {type(value).__name__})")

        # Example of accessing nested structure if you had one
        # if 'reconstruction' in config and isinstance(config['reconstruction'], dict):
        #     print(f"\nReconstruction Algorithm: {config['reconstruction'].get('algorithm', 'N/A')}")
        #     print(f"FBP Filter: {config['reconstruction'].get('fbp_filter', 'N/A')}")

        # Example direct access
        print(f"\nDataset Name: {config.get('dataset_name', 'Not Set')}")
        print(f"Log Level: {config.get('log_level', 'INFO')}")

    else:
        logger.error("Failed to load configuration.")

    # Optional: Clean up dummy file if created
    # if 'dummy_config_content' in locals(): # Check if we created the dummy file
    #     try:
    #         default_config_file.unlink()
    #         logger.info(f"Removed dummy config file: {default_config_file}")
    #     except Exception as e:
    #         logger.error(f"Failed to remove dummy config file: {e}")


    logger.info("\nConfig tests finished.")

