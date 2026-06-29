"""
get_RBF_Scores.py

This script performs Step 3: Scores RBF, calculating comprehensive performance metrics for SVM RBF models.
It loads results from SFS (Step 1) and combination search (Step 2), computes advanced metrics like MCC, F1, 
specificity, NPV, PPV, and likelihood ratios from confusion matrices, and saves detailed performance summaries.

Usage (as a function):
    get_RBF_scores(
        sfs_results_file,
        combination_results_file,
        output_file='Scores_RBF_older.xlsx',
        logger=None
    )
"""
import pandas as pd
import numpy as np
import os
from ast import literal_eval
from numpy import sqrt
import logging

# Import existing metric functions from utils
from ..utils.utils_metrics import (
    get_MCC_from_cm, get_specificity_from_cm, get_NPV, get_PPV, 
    get_f1_from_cm, get_likelihood_ratios_from_cm, get_ppv_from_cm, 
    get_npv_from_cm, parse_confusion_matrix
)

def get_RBF_scores(
    sfs_results_file,
    combination_results_file,
    output_file='Scores_RBF_older.xlsx',
    logger=None
):
    """
    Calculate comprehensive performance metrics for SVM RBF models.
    
    Args:
        sfs_results_file (str): Path to SFS results Excel file (from Step 1)
        combination_results_file (str): Path to combination results Excel file (from Step 2)
        output_file (str): Output Excel file path
        logger: Logger instance for logging
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading SFS results from: {sfs_results_file}")
    logger.info(f"Loading combination results from: {combination_results_file}")

    # Read the Excel files
    excel_step1 = pd.ExcelFile(sfs_results_file)
    excel_step2 = pd.ExcelFile(combination_results_file)
    
    score_rbf = {}

    for sheet_name in excel_step1.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")
        
        # Read spreadsheets for this sheet
        df_step1 = pd.read_excel(excel_step1, sheet_name=sheet_name)
        df_step2 = pd.read_excel(excel_step2, sheet_name=sheet_name)

        # Get significant SFS models
        significant_sfs_models = list(df_step2['Feature_idx'].iloc)
        
        # Get feature counts for each combination
        length_feat = list(df_step2['#Features'].iloc[:])
        
        # Get model names (row indices)
        name_models = list(df_step2.iloc[:, 0])
        
        # Get all confusion matrices
        matrix_list = list(df_step2['Confusion_Matrix'].iloc[:])
        counter = 0

        # DataFrame for results
        df_results = pd.DataFrame(columns=[
            'Name_Model', '#Features', 'C_Value', 'Gamma_Value', 'Feature_idx', 'tol',
            'CV_accuracy', 'Test_Accuracy', 'Specificity', 'Sensitivity', 'NPV', 'PPV',
            'Likelihood_Ratio', 'F1', 'MCC', 'Confusion_Matrix', 'Features'
        ])

        for i in significant_sfs_models:
            # Get feature indices for this SFS model
            features = np.array(literal_eval(str(df_step1.iloc[:, -1][i])))
            
            # Get feature combination for this model
            feat_comb = features[:length_feat[counter]]
            
            # Parse confusion matrix
            matrix = parse_confusion_matrix(matrix_list[counter])
            tn, fp, fn, tp = matrix.ravel()

            # Calculate metrics using confusion matrix components
            specificity_val = get_specificity_from_cm(tn, fp, fn, tp)
            npv_val = get_npv_from_cm(tn, fn)
            ppv_val = get_ppv_from_cm(tp, fp)
            mcc_val = get_MCC_from_cm(tn, fp, fn, tp)
            f1_val = get_f1_from_cm(tp, fp, fn, tn)
            lh_val = get_likelihood_ratios_from_cm(tp, fp, fn, tn)

            # Save to results DataFrame
            df_results = pd.concat([df_results,
                pd.DataFrame({
                    'Name_Model': [name_models[counter]],
                    '#Features': [df_step2.iloc[counter, 4]],
                    'C_Value': [df_step2.iloc[counter, 5]],
                    'Gamma_Value': [df_step2.iloc[counter, 6]],
                    'CV_split': [df_step2.iloc[counter, 10]],
                    'Random_State': [df_step2.iloc[counter, 11]],
                    'Feature_idx': [i],
                    'tol': ['RBF'],
                    'CV_accuracy': [df_step2.iloc[counter, 1]],
                    'Test_Accuracy': [df_step2.iloc[counter, 2]],
                    'Specificity': [specificity_val],
                    'Sensitivity': [df_step2.iloc[counter, 3]],
                    'NPV': [npv_val],
                    'PPV': [ppv_val],
                    'Likelihood_Ratio': [lh_val],
                    'F1': [f1_val],
                    'MCC': [mcc_val],
                    'Confusion_Matrix': [df_step2.iloc[counter, 8]],
                    'Features': [str(feat_comb)]
                })], ignore_index=True)

            counter += 1

        score_rbf[sheet_name] = df_results

    # Save results to Excel file
    logger.info(f"Saving results to: {output_file}")
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df_results in score_rbf.items():
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)
    
    logger.info("RBF scores calculation completed successfully")
    return score_rbf

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 3: Scores RBF - Calculate comprehensive performance metrics for SVM RBF models.")
    parser.add_argument('--sfs', required=True, help='Path to SFS results Excel file (from Step 1)')
    parser.add_argument('--combinations', required=True, help='Path to combination results Excel file (from Step 2)')
    parser.add_argument('--output', default='Scores_RBF_older.xlsx', help='Output Excel file path')
    
    args = parser.parse_args()
    
    get_RBF_scores(
        sfs_results_file=args.sfs,
        combination_results_file=args.combinations,
        output_file=args.output
    ) 