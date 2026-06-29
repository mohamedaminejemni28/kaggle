import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# define the project name and package name
project_name = "gaitML"
package_name = "gaitml"  # lowercase for package name

list_of_files = [
    # GitHub workflows
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",
    
    # Source code structure
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/data/__init__.py",
    f"src/{package_name}/data/data_loader.py",
    f"src/{package_name}/data/data_processor.py",
    f"src/{package_name}/features/__init__.py",
    f"src/{package_name}/features/feature_engineering.py",
    f"src/{package_name}/features/feature_selection.py",
    f"src/{package_name}/models/__init__.py",
    f"src/{package_name}/models/train_model.py",
    f"src/{package_name}/models/predict_model.py",
    f"src/{package_name}/models/evaluate_model.py",
    f"src/{package_name}/visualization/__init__.py",
    f"src/{package_name}/visualization/visualize.py",
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/utils/helpers.py",
    
    # Data directories
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    
    # Notebooks
    "notebooks/1.0-data-exploration.ipynb",
    "notebooks/2.0-feature-engineering.ipynb",
    "notebooks/3.0-model-training.ipynb",
    "notebooks/4.0-model-evaluation.ipynb",
    
    # Tests
    "tests/__init__.py",
    "tests/test_data.py",
    "tests/test_features.py",
    "tests/test_models.py",
    "tests/test_utils.py",
    
    # Configuration files
    "configs/config.yaml",
    "configs/hyperparameters.yaml",
    "configs/model_config.yaml",
    
    # Scripts
    "scripts/train.py",
    "scripts/predict.py",
    "scripts/evaluate.py",
    "scripts/data_preprocessing.py",
    
    # Documentation
    "docs/api.md",
    "docs/user_guide.md",
    "docs/installation.md",
    
    # Project configuration files
    # "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    # "setup.py",
    "dvc.yaml",
    # "params.yaml",
    # ".env.example",
    # "Makefile",
    "README.md",
    
    # # Model artifacts directory
    # "models/.gitkeep",
    
    # Logs directory
    "logs/.gitkeep",
    
    # Reports directory
    "reports/figures/.gitkeep",
    "reports/tables/.gitkeep"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")
