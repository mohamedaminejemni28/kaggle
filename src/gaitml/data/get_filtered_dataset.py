"""
Review Feature Names

This script verifies that all Joint Angle and Temporal-Spatial Feature Names are present
in both variant and invariant datasets, filters the datasets to include only these features,
and saves the filtered datasets.

"""

import pandas as pd
import numpy as np
from pathlib import Path

# Get the logger from the package
from gaitml import logger

def load_feature_names(file_path: str) -> pd.DataFrame:
    """
    Load feature names from Excel spreadsheet.
    
    Args:
        file_path (str): Path to the Excel file containing feature names
        
    Returns:
        pd.DataFrame: DataFrame containing Joint Angle and Temporal-Spatial features
    """
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Successfully loaded feature names from {file_path}")
        logger.info(f"Feature names shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        raise


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from Excel file.
    
    Args:
        file_path (str): Path to the Excel file containing the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Successfully loaded dataset from {file_path}")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def extract_feature_names(df_ja_ts: pd.DataFrame) -> tuple:
    """
    Extract Joint Angle and Temporal-Spatial feature names from the reference DataFrame.
    
    Args:
        df_ja_ts (pd.DataFrame): DataFrame containing feature names
        
    Returns:
        tuple: Lists of feature names for variant and invariant datasets
    """
    # Combine the two columns into a single list and remove NaN values
    combined_features = pd.concat([
        df_ja_ts['Joint Angle Features'], 
        df_ja_ts['Temporal-Spatial Features']
    ])
    
    # Drop NaN values
    ja_ts_features = combined_features.dropna().tolist()
    
    # Define index columns for each dataset type
    index_columns_variant = ['c', 'Index', 'Group', 'Side']
    index_columns_invariant = ['Participant', 'Index', 'Group', 'Side']
    
    # Create feature lists for each dataset
    ja_ts_features_variant = index_columns_variant + ja_ts_features
    ja_ts_features_invariant = index_columns_invariant + ja_ts_features
    
    logger.info(f"Extracted {len(ja_ts_features)} Joint Angle and Temporal-Spatial features")
    logger.info(f"Variant dataset will have {len(ja_ts_features_variant)} columns")
    logger.info(f"Invariant dataset will have {len(ja_ts_features_invariant)} columns")
    
    return ja_ts_features_variant, ja_ts_features_invariant


def filter_dataset(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Filter dataset to include only specified features.
    
    Args:
        df (pd.DataFrame): Original dataset
        feature_list (list): List of feature names to include
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    # Filter columns to include only those in the feature list
    filtered_df = df.loc[:, df.columns.isin(feature_list)]
    
    logger.info(f"Filtered dataset shape: {filtered_df.shape}")
    # logger.info(f"Columns in filtered dataset: {list(filtered_df.columns)}")
    
    return filtered_df


def save_filtered_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save filtered dataset to Excel file.
    
    Args:
        df (pd.DataFrame): Dataset to save
        output_path (str): Output file path
    """
    try:
        df.to_excel(output_path, index=False)
        logger.info(f"Successfully saved filtered dataset to {output_path}")
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        raise


def get_filtered_datasets(feature_names_file, variant_file, invariant_file, variant_output, invariant_output):
    """
    Main function to execute the feature review process.
    """
    logger.info("Starting feature review process...")

    # Output file paths
    try:
        # Step 1: Load feature names
        logger.info("Loading feature names...")
        df_ja_ts = load_feature_names(feature_names_file)
        
        # Step 2: Load datasets
        logger.info("Loading variant dataset...")
        df_variant = load_dataset(variant_file)
        
        logger.info("Loading invariant dataset...")
        df_invariant = load_dataset(invariant_file)
        
        # Step 3: Extract feature names
        logger.info("Extracting feature names...")
        ja_ts_features_variant, ja_ts_features_invariant = extract_feature_names(df_ja_ts)
        
        # Step 4: Filter datasets
        logger.info("Filtering variant dataset...")
        df_variant_filtered = filter_dataset(df_variant, ja_ts_features_variant)
        
        logger.info("Filtering invariant dataset...")
        df_invariant_filtered = filter_dataset(df_invariant, ja_ts_features_invariant)
        
        # Step 5: Save filtered datasets
        logger.info("Saving filtered datasets...")
        save_filtered_dataset(df_variant_filtered, variant_output)
        save_filtered_dataset(df_invariant_filtered, invariant_output)
        
        logger.info("Feature review process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature review process: {e}")
        raise


if __name__ == "__main__":
    # File paths
    main_dir_path = "F:/python_venv/Gait-ML-Shap/"
    feature_names_file = main_dir_path + "data/raw/ISB Abstract - Joint Angles and TS Variable Names.xlsx"
    variant_file = main_dir_path + "data/raw/2024 ALL Collected Speeds Young and Older Adults ALL Features - FINAL AUGUST 20.xlsx"
    invariant_file = main_dir_path + "data/raw/2024 ALL Post-Processing Speeds Young and Older Adults ALL Features - FINAL AUGUST 20.xlsx"
    
    variant_output = "F:/python_venv/Gait-ML-Shap/data/processed/2024 ALL Collected Speeds Young and Older Adults JA and TS features - FINAL AUGUST 20.xlsx"
    invariant_output = "F:/python_venv/Gait-ML-Shap/data/processed/2024 ALL Post-Processing Speeds Young and Older Adults JA and TS features - FINAL AUGUST 20.xlsx"    

    # Run the feature review process
    get_filtered_datasets(feature_names_file, variant_file, invariant_file, variant_output, invariant_output)
