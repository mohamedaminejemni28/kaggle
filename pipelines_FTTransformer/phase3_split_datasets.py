import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gaitml import logger
from gaitml.utils.helpers import read_yaml
from gaitml.data.create_train_val_test_split import create_train_test_split

def main():
    logger.info(">>>>>>>>>> Starting Phase 3: Splitting Datasets <<<<<<<<<<")

    # Read config file
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    # Extract paths from config
    input_file = config["split_datasets"]["input_file"]
    output_train_file = config["split_datasets"]["output_train_file"]
    output_test_file = config["split_datasets"]["output_test_file"]

    create_train_test_split(input_excel_file = input_file,                            
                            output_train_file = output_train_file,
                            output_test_file = output_test_file,
                            group_column = "Group",
                            test_size = 0.2,
                            stratify = True,
                            random_state = 0)   

    logger.info(">>>>>>>>>> Completed Phase 3: Splitting Datasets <<<<<<<<<<")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}") 
        raise e

