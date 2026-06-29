"""
Create Train/Test Split for each Sheet in the dataset.

This script splits each sheet in an Excel file into train and test sets,
stratified by the 'Group' column, and saves the results to new Excel files.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from gaitml import logger

def create_train_test_split(
    input_excel_file: str,
    output_train_file: str,
    output_test_file: str,
    group_column: str = "Group",
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 0,
):
    """
    Split each sheet in the Excel file into train and test sets.

    Args:
        input_excel_file (str): Path to the dataset (input Excel file).
        output_train_file (str): Path to save the train set (output Excel file).
        output_test_file (str): Path to save the test set (output Excel file).
        group_column (str): Name of the column to split by.
        test_size (float): Proportion of the test set.
        stratify (bool): Whether to stratify the split by the group column.
        random_state (int): Random seed for reproducibility.
    """
    logger.info(f"Loading Dataset Excel file: {input_excel_file}")
    excel_in = pd.ExcelFile(input_excel_file)

    train_sheets = {}
    test_sheets = {}

    for sheet_name in excel_in.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)
        if group_column not in df.columns:
            logger.error(f"Column '{group_column}' not found in {sheet_name}.")
            raise ValueError(f"Column '{group_column}' not found in {sheet_name}.")

        X = df
        y = df[group_column]

        if stratify:
            training_data, testing_data, _, _ = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
        else:
            training_data, testing_data, _, _ = train_test_split(X, test_size=test_size, random_state=random_state)
            
        train_sheets[sheet_name] = training_data
        test_sheets[sheet_name] = testing_data
        logger.info(f"training_data shape: {training_data.shape}, tesing_data shape: {testing_data.shape}")

    logger.info(f"Saving train sets to {output_train_file}")
    with pd.ExcelWriter(output_train_file) as writer:
        for sheet_name, training_data in train_sheets.items():
            training_data.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Saving test sets to {output_test_file}")
    with pd.ExcelWriter(output_test_file) as writer:
        for sheet_name, testing_data in test_sheets.items():
            testing_data.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("Train/test split complete.")

if __name__ == "__main__":
    # Example usage (update file names as needed)
    main_dir_path = "F:/python_venv/Gait-ML-Shap/"
    input_file = main_dir_path + "data/processed/ALL FEATURES OLDER - Pes Planus and Control NH Correlated features JUNE 25.xlsx"
    output_train_file = main_dir_path + "data/processed/ALL FEATURES OLDER - Pes Planus and Control NH Correlated features JUNE 25 Train.xlsx"
    output_test_file = main_dir_path + "data/processed/ALL FEATURES OLDER - Pes Planus and Control NH Correlated features JUNE 25 Test.xlsx"

    create_train_test_split(
        input_excel_file=input_file,
        output_train_file=output_train_file,
        output_test_file=output_test_file,
        group_column="Group",
        test_size=0.2,
        random_state=0,
    )
