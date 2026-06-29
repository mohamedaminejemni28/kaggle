"""
SHAP Data Analysis Script

This script performs SHAP (SHapley Additive exPlanations) analysis on SVM models
to understand feature importance and model interpretability.

Author: roqui
Modified by: Assistant
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import shap
from sklearn.preprocessing import StandardScaler
import argparse
import os
from pathlib import Path


def load_model_parameters(excel_file_path):
    """
    Load model parameters from Excel file.
    
    Args:
        excel_file_path (str): Path to the Excel file containing model parameters
        
    Returns:
        pd.DataFrame: DataFrame containing model parameters
    """
    paramaters_EXCEL = pd.ExcelFile(excel_file_path)
    
    dfs = []
    for sheet in paramaters_EXCEL.sheet_names:
        df = pd.read_excel(paramaters_EXCEL, sheet_name=sheet)
        df['SPEED_MODEL'] = sheet
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def load_datasets(train_path, test_path):
    """
    Load training and testing datasets.
    
    Args:
        train_path (str): Path to training data Excel file
        test_path (str): Path to testing data Excel file
        
    Returns:
        tuple: (train_excel, test_excel) ExcelFile objects
    """
    train_EXCEL = pd.ExcelFile(train_path)
    test_EXCEL = pd.ExcelFile(test_path)
    return train_EXCEL, test_EXCEL


def prepare_data(df, group_mapping={'Young': 0, 'Older': 1}):
    """
    Prepare data by extracting features and labels.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_mapping (dict): Mapping for group labels
        
    Returns:
        tuple: (Y, segments) where Y is labels and segments is feature dictionary
    """
    df["Group"] = df["Group"].replace(group_mapping)
    Y = df.loc[:, "Group"].values
    
    segments = {
        'All_Data': df.loc[:, 'Pelv_Angle_Y_MAX_SW': 'Hip_Angle_Z_OHS']
    }
    
    return Y, segments


def extract_selected_features(segments, feature_indices):
    """
    Extract selected features based on feature indices.
    
    Args:
        segments (dict): Dictionary containing feature segments
        feature_indices (list): List of feature indices to select
        
    Returns:
        pd.DataFrame: DataFrame with selected features
    """
    sfs_feature_names = np.array([int(i) for i in feature_indices])
    return segments.iloc[:, sfs_feature_names]


def perform_shap_analysis(X_train, Y_train, X_test, Y_test, feature_names, 
                         model_params, output_name, output_dir="."):
    """
    Perform SHAP analysis on the model.
    
    Args:
        X_train (pd.DataFrame): Training features
        Y_train (np.array): Training labels
        X_test (pd.DataFrame): Testing features
        Y_test (np.array): Testing labels
        feature_names (list): List of feature names
        model_params (dict): Model parameters (C, gamma)
        output_name (str): Name for output files
        output_dir (str): Directory to save outputs
        
    Returns:
        pd.DataFrame: DataFrame containing SHAP values
    """
    # Apply scaling
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    # Train SVM model
    svm = SVC(kernel='rbf', C=model_params['C'], gamma=model_params['gamma'])
    svm.fit(X_train_scaled, Y_train)
    
    print(f"Model Accuracy: {svm.score(X_test_scaled, Y_test):.4f}")
    
    # SHAP analysis
    svm_explainer = shap.KernelExplainer(svm.predict, X_test_scaled, 
                                        feature_names=feature_names)
    shap_values = svm_explainer.shap_values(X_test_scaled)
    
    # Create summary plot
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, 
                     plot_type="bar", max_display=50, show=False)
    plot_path = os.path.join(output_dir, f"{output_name}_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return shap_values


def create_shap_dataframe(shap_values, feature_names, metadata_df):
    """
    Create DataFrame with SHAP values and metadata.
    
    Args:
        shap_values (np.array): SHAP values
        feature_names (list): Feature names
        metadata_df (pd.DataFrame): DataFrame with metadata columns
        
    Returns:
        pd.DataFrame: DataFrame with SHAP values and metadata
    """
    df_SHAP_Vals = pd.DataFrame(shap_values, columns=feature_names)
    df_SHAP_Vals = pd.concat([metadata_df[['Participant', 'Index', 'Side', 'Group']], 
                             df_SHAP_Vals], axis=1)
    
    # Calculate sum of SHAP values
    df_SHAP_Vals['SUM'] = df_SHAP_Vals[feature_names].sum(axis=1)
    
    # Find most contributing features
    min_cols = df_SHAP_Vals[feature_names].idxmin(axis=1)
    max_cols = df_SHAP_Vals[feature_names].idxmax(axis=1)
    df_SHAP_Vals['Most Contributing Feat'] = np.where(df_SHAP_Vals['Group'] == 0, 
                                                     min_cols, max_cols)
    df_SHAP_Vals['Most Contributing Val'] = df_SHAP_Vals.apply(
        lambda row: row[row['Most Contributing Feat']], axis=1)
    
    return df_SHAP_Vals


def get_SHAP_data(scores_file, train_file, test_file, output_file, output_dir="."):
    """
    Main function to perform SHAP analysis.
    
    Args:
        scores_file (str): Path to scores Excel file
        train_file (str): Path to training data Excel file
        test_file (str): Path to testing data Excel file
        output_file (str): Output Excel file name
        output_dir (str): Directory to save outputs
    """
    print("Loading model parameters...")
    models_parameters = load_model_parameters(scores_file)
    
    print("Loading datasets...")
    train_EXCEL, test_EXCEL = load_datasets(train_file, test_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    df_SHAP = {}
    
    for i in range(len(models_parameters)):
        print(f"\nProcessing model {i+1}/{len(models_parameters)}...")
        
        # Extract model parameters
        SHEET_SPEED = models_parameters['SPEED_MODEL'][i]
        C_VAL = models_parameters['C_Value'][i]
        R_VAL = models_parameters['Gamma_Value'][i]
        FEATS = models_parameters['Features'][i].strip("[]").split()
        
        SHAP_NAME = f"SHAP_{models_parameters['Name_Model'][i]}"
        
        # Load training data
        df_YO_train = pd.read_excel(train_EXCEL, sheet_name=SHEET_SPEED)
        Y_YO_train, segments_YO_train = prepare_data(df_YO_train)
        
        # Load testing data
        df_YO_test = pd.read_excel(test_EXCEL, sheet_name=SHEET_SPEED)
        Y_YO_test, segments_YO_test = prepare_data(df_YO_test)
        
        # Merge datasets
        df_All_set = pd.concat([df_YO_train, df_YO_test], ignore_index=True)
        Y_YO_All = df_All_set.loc[:, "Group"].values
        segments_ALL = {
            'AllF': df_All_set.loc[:, 'Pelv_Angle_Y_MAX_SW': 'Hip_Angle_Z_OHS']
        }
        
        # Extract selected features
        df_segment_train = extract_selected_features(segments_YO_train["All_Data"], FEATS)
        df_segment_test = extract_selected_features(segments_YO_test["All_Data"], FEATS)
        df_segment_all = extract_selected_features(segments_ALL["AllF"], FEATS)
        
        feature_names = df_segment_all.columns.values.tolist()
        print(f"Selected {len(feature_names)} features")
        
        # Perform SHAP analysis
        model_params = {'C': C_VAL, 'gamma': R_VAL}
        shap_values = perform_shap_analysis(
            df_segment_train, Y_YO_train, df_segment_all, Y_YO_All,
            feature_names, model_params, SHAP_NAME, output_dir
        )
        
        # Create SHAP DataFrame
        df_SHAP_Vals = create_shap_dataframe(shap_values, feature_names, df_All_set)
        
        shap_name = f"{SHEET_SPEED}_{models_parameters['Name_Model'][i]}"
        df_SHAP[shap_name] = df_SHAP_Vals
    
    # Save results
    output_path = os.path.join(output_dir, output_file)
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in df_SHAP.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\nSHAP analysis completed. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform SHAP analysis on SVM models")
    parser.add_argument("--scores", required=True, 
                       help="Path to scores Excel file")
    parser.add_argument("--train", required=True, 
                       help="Path to training data Excel file")
    parser.add_argument("--test", required=True, 
                       help="Path to testing data Excel file")
    parser.add_argument("--output", default="SHAP_analysis.xlsx", 
                       help="Output Excel file name")
    parser.add_argument("--output-dir", default=".", 
                       help="Directory to save outputs")
    
    args = parser.parse_args()
    
    get_SHAP_data(args.scores, args.train, args.test, args.output, args.output_dir) 