import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.data.get_filtered_dataset import get_filtered_datasets

def main():
    logger.info(">>>>>>>>>> Starting Phase 1: Reviewing Features <<<<<<<<<<")

    # Read config file
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    # Extract paths from config
    feature_names_file = config["data"]["feature_names_file"]
    variant_file = config["data"]["variant_file"]
    invariant_file = config["data"]["invariant_file"]
    variant_output = config["data"]["variant_output"]
    invariant_output = config["data"]["invariant_output"]

    get_filtered_datasets(feature_names_file, variant_file, invariant_file, variant_output, invariant_output)

    logger.info(">>>>>>>>>> Completed Phase 1: Reviewing Features <<<<<<<<<<")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e

