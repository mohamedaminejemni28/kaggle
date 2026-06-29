"""
Remove Correlated Features

This script removes highly correlated features from all sheets in a given Excel file.
"""

import pandas as pd
from gaitml import logger


def remove_correlated_features(
    correlated_features_file: str,
    input_excel_file: str,
    output_excel_file: str
):
    """
    Remove highly correlated features from all sheets in an Excel file.

    Args:
        correlated_features_file (str): Path to Excel file with features to drop (column: Feature_names)
        input_excel_file (str): Path to input Excel file with multiple sheets
        output_excel_file (str): Path to save the filtered Excel file
    """
    ## Load Excel File with the features correlated list
    logger.info(f"Loading correlated features from {correlated_features_file}")
    df_corr = pd.read_excel(correlated_features_file, header=None)
    df_corr.rename(columns={0: "Feature_names"}, inplace=True)
    columns_to_drop = df_corr["Feature_names"].tolist()

    logger.info(f"Loading Excel file: {input_excel_file}")
    excel_in = pd.ExcelFile(input_excel_file)

    filter_sheets = {}

    for sheet_name in excel_in.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)
        missing_columns = [col for col in columns_to_drop if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {sheet_name}: {missing_columns}")
        df_filtered = df.drop(columns=columns_to_drop, errors="ignore")
        filter_sheets[sheet_name] = df_filtered

    logger.info(f"Saving filtered sheets to {output_excel_file}")
    with pd.ExcelWriter(output_excel_file) as writer:
        for sheet_name, df_filtered in filter_sheets.items():
            df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == "__main__":

    main_dir_path = "F:/python_venv/Gait-ML-Shap/"
    
    highly_correlated_features_file = main_dir_path + "data/raw/Highly correlated features.xlsx"
    input_excel_file = main_dir_path + "data/raw/ALL FEATURES OLDER - Pes Planus and Control JUNE 25.xlsx"
    output_excel_file = main_dir_path + "data/processed/ALL FEATURES OLDER - Pes Planus and Control NH Correlated features JUNE 25.xlsx"

    remove_correlated_features(
        highly_correlated_features_file,
        input_excel_file,
        output_excel_file
    )
