import yaml
from .. import logger


# read yaml file
def read_yaml(path_to_yaml: str) -> dict:
    logger.info(f"Reading YAML file from {path_to_yaml}")
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logger.info(f"YAML file loaded successfully from {path_to_yaml}")
    return content



