import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.data.remove_correlated_features import remove_correlated_features

def main():
    logger.info(">>>>>>>>>> Starting Phase 2: Removing Correlated Features <<<<<<<<<<")

    # Read config file
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)
    
    # Extract paths from config
    highly_correlated_features_file = config["remove_correlated_features"]["highly_correlated_features_file"]
    input_excel_file = config["remove_correlated_features"]["input_excel_file"]
    output_excel_file = config["remove_correlated_features"]["output_excel_file"]

    remove_correlated_features(highly_correlated_features_file, input_excel_file, output_excel_file)

    logger.info(">>>>>>>>>> Completed Phase 2: Removing Correlated Features <<<<<<<<<<<")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
