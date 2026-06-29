# GaitML - Machine Learning Project for Gait Analysis

A comprehensive machine learning project template for gait analysis using modern Python practices and cookiecutter data science conventions.

## Project Structure

```
gaitML/
├── src/
│   └── gaitml/                    # Main package source code
│       ├── __init__.py
│       ├── data/                  # Data handling modules
│       │   ├── __init__.py
│       │   ├── data_loader.py     # Data loading utilities
│       │   ├── data_processor.py  # Data preprocessing
│       │   ├── get_filtered_dataset.py  # Feature filtering functionality
│       │   ├── remove_correlated_features.py  # Remove correlated features
│       │   └── create_train_val_test_split.py # Train/test split
│       ├── features/              # Feature engineering and selection
│       │   ├── __init__.py
│       │   ├── feature_engineering.py
│       │   ├── feature_selection.py
│       │   └── sequential_feature_selection.py   # SFS logic for Phase 4
│       ├── models/                # ML model components
│       │   ├── __init__.py
│       │   ├── train_model.py     # Model training
│       │   ├── predict_model.py   # Model prediction
│       │   └── evaluate_model.py  # Model evaluation
│       ├── visualization/         # Plotting and visualization
│       │   ├── __init__.py
│       │   └── visualize.py
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           └── helpers.py
├── pipelines/                     # Data processing pipelines
│   ├── phase1_reviewfeatures.py           # Phase 1: Feature filtering pipeline
│   ├── phase2_removecorrelated_features.py # Phase 2: Remove correlated features
│   ├── phase3_split_datasets.py           # Phase 3: Train/test split
│   ├── phase4_1_SFS.py                    # Phase 4.1: Sequential Feature Selection (SFS)
│   ├── phase4_2_model_combinations.py     # Phase 4.2: Model combinations
│   ├── phase4_3_scoresRBF.py              # Phase 4.3: RBF scores
│   └── phase4_4_SHAP_data.py              # Phase 4.4: SHAP data analysis
├── data/                          # Data storage
│   ├── raw/                       # Raw data files (CSV, JSON, etc.)
│   ├── processed/                 # Processed/cleaned data
│   └── external/                  # External data sources
├── notebooks/                     # Jupyter notebooks
├── configs/                       # Configuration files
│   ├── config.yaml               # Main configuration
│   ├── hyperparameters.yaml      # Model hyperparameters
│   └── model_config.yaml         # Model-specific config
├── logs/                          # Log files
├── .github/                       # GitHub workflows
├── requirements.txt               # Production dependencies
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
├── extras/                        # Legacy, unused, or reference materials
```

## Where to Put Different Types of Files

### Data Files
- **Raw data**: `data/raw/` - Original, unprocessed data files
- **Processed data**: `data/processed/` - Cleaned and preprocessed data
- **External data**: `data/external/` - Data from external sources
- **Large files**: Use DVC for version control (add to `.gitignore`)

### Analysis & Exploration
- **Jupyter notebooks**: `notebooks/` - Use numbered prefixes for workflow order
- **Research notebooks**: `notebooks/` - Keep exploratory work organized
- **Final analysis**: `notebooks/` - Production-ready analysis

### Code Organization
- **Core functionality**: `src/gaitml/` - Main package code
- **Data handling**: `src/gaitml/data/` - Data loading and processing
- **Feature engineering**: `src/gaitml/features/` - Feature creation and selection (including SFS logic)
- **ML models**: `src/gaitml/models/` - Training, prediction, evaluation
- **Visualization**: `src/gaitml/visualization/` - Plotting and charts
- **Utilities**: `src/gaitml/utils/` - Helper functions and tools

### Pipelines
- **Data processing pipelines**: `pipelines/` - End-to-end data processing workflows
- **Phase 1**: `phase1_reviewfeatures.py` - Feature filtering and dataset preparation
- **Phase 2**: `phase2_removecorrelated_features.py` - Remove highly correlated features from datasets
- **Phase 3**: `phase3_split_datasets.py` - Split datasets into train and test sets
- **Phase 4**: `phase4_1_SFS.py` - Sequential Feature Selection (SFS) on train set (uses code in `src/gaitml/features/sequential_feature_selection.py`)
- **Future phases**: Additional pipeline stages for feature engineering, model training, etc.

### Configuration
- **Main config**: `configs/config.yaml` - General project settings
- **Model params**: `configs/hyperparameters.yaml` - ML model parameters
- **Model config**: `configs/model_config.yaml` - Model-specific settings
- **Environment vars**: `.env` (not tracked) - Sensitive configuration

### Testing
- **Unit tests**: `tests/` - Test files matching source structure
- **Test data**: `tests/fixtures/` - Test data files
- **Integration tests**: `tests/integration/` - End-to-end tests

### Documentation
- **API docs**: `docs/api.md` - Code documentation
- **User guides**: `docs/user_guide.md` - How to use the project
- **Installation**: `docs/installation.md` - Setup instructions
- **Project docs**: `README.md` - Project overview

### Scripts & Automation
- **Training scripts**: `scripts/train.py` - Model training
- **Prediction scripts**: `scripts/predict.py` - Making predictions
- **Evaluation scripts**: `scripts/evaluate.py` - Model evaluation
- **Data scripts**: `scripts/data_preprocessing.py` - Data pipeline

### Outputs & Results
- **Model artifacts**: `models/` - Trained model files
- **Logs**: `logs/` - Application and training logs
- **Reports**: `reports/figures/` and `reports/tables/` - Generated outputs
- **Plots**: `reports/figures/` - Saved visualizations

### CI/CD & Workflows
- **GitHub Actions**: `.github/workflows/` - Automated workflows
- **DVC pipelines**: `dvc.yaml` - Data pipeline configuration
- **Parameters**: `params.yaml` - DVC parameters

## Extras Directory

Unused, legacy, or experimental files and folders have been moved to the `extras/` directory. This helps keep the main project structure clean and focused on active development.

- **Purpose:** The `extras/` directory contains scripts, notebooks, or data that are not part of the current pipeline but may be useful for reference, archival, or future work.
- **What you might find:**
  - Old versions of scripts
  - Deprecated data processing code
  - Experimental notebooks
  - Miscellaneous resources

Feel free to explore the `extras/` directory if you are looking for legacy code or additional resources that are not part of the main workflow.

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- DVC (for data version control)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gaitML
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

4. Set up DVC (if using data version control):
   ```bash
   dvc init
   dvc remote add origin <dvc-remote-url>
   ```

### Usage

#### Data Processing Pipelines
1. **Phase 1 - Feature Review**: Run the feature filtering pipeline:
   ```bash
   python pipelines/phase1_reviewfeatures.py
   ```
   - Loads feature names from reference Excel file
   - Filters variant and invariant datasets to include only Joint Angle and Temporal-Spatial features
   - Saves filtered datasets for further processing

2. **Phase 2 - Remove Correlated Features**: Run the correlated feature removal pipeline:
   ```bash
   python pipelines/phase2_removecorrelated_features.py
   ```
   - Identify highly correlated features between groups
   - Removes the highly correlated features from the 'All Features' sheets in the Excel file
   - Saves the cleaned datasets (without the highly correlated features) for further processing

3. **Phase 3 - Train/Test Split**: Run the train/test split pipeline:
   ```bash
   python pipelines/phase3_split_datasets.py
   ```
   - Splits each sheet in the Excel file into train and test sets (stratified by 'Group')
   - Saves the train and test sets for all sheets into separate Excel files

4. **Phase 4 - Step 1: Sequential Feature Selection (SFS)**: Run the SFS pipeline:
   ```bash
   python pipelines/phase4_1_SFS.py
   ```
   - Performs sequential feature selection (SFS) using SVM (RBF) for each sheet in the train set Excel file
   - Stores SFS results (accuracy, C, gamma, number of features, feature order) for each parameter combination
   - Saves results for each sheet to a new Excel file
   - The SFS logic is implemented in `src/gaitml/features/sequential_feature_selection.py`
   - Generates an output file called '..._ML_Step_1.xlsx'

5. **Phase 4 - Step 2: Model Combinations**: Generate and evaluate model combinations:
   ```bash
   python pipelines/phase4_2_model_combinations.py
   ```
   - Generates and evaluates all possible model/feature combinations based on SFS results
   - Saves combination results (e.g., accuracy, selected features) to Excel
   - Replaces each group name with a number (i.e., '0' and '1')
   - Generates an output file called '..._Step_2_Results.xlsx'
   - Before proceeding to Step 3, the 'Step_2_Results' file needs to be filitered, depending on the reserach question
   - The filtering process is as followed:\
     a) Rename the Column ‘A’ (the index number column) “Name_Model”.\
     b) Create a table (with headers) within the spreadsheet.\
     c) Filter the ‘CV_Accuracy’ column by using the ‘Number_Filter’ command. Select ‘Greater Than or Equal to’ filter and set the value to 0.90.\
     d) Filter the ‘CV_Accuracy’ column by using the ‘Number_Filter’ command. Select ‘Greater Than or Equal to’ filter and set the value to 0.85.\
     e) Filter the ‘CV_Accuracy’ column by using the ‘Number_Filter’ command. Select ‘Greater Than or Equal to’ filter and set the value to 0.85.\
     f) Filter the ‘#Features’ column based on the desired number of features (e.g., 5 and 6 features).\
     g) Sort the data within the table based on the columns 1) CV_Accuracy (largest to smallest), then by 2) Test_Accuracy (largest to smallest) and finally by 3) Sensitivity (ALL largest to smallest).

6. **Phase 4 - Step 3: RBF Scores**: Calculate and summarize SVM RBF model scores:
   ```bash
   python pipelines/phase4_3_scoresRBF.py
   ```
   - Calculates detailed performance metrics (accuracy, MCC, specificity, etc.) for all model combinations
   - Saves results to an Excel file for further analysis

7. **Phase 4 - Step 4: SHAP Data Analysis**: Run the SHAP analysis pipeline:
   ```bash
   python pipelines/phase4_4_SHAP_data.py
   ```
   - Analyzes feature importance and model interpretability for SVM models using SHAP
   - Generates summary plots and Excel files with SHAP values and metadata for each model

#### Traditional Workflow
1. **Data Exploration**: Start with `
