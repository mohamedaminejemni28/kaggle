"""
get_combination_list_with_accuracy.py

This script performs Step 2: Combinations, evaluating SVM models on various feature subsets and hyperparameter combinations.
It loads train and test Excel files, iterates over feature sets and SVM parameters, computes cross-validation and test accuracy, sensitivity, and other metrics, and saves the results for each sheet to an output Excel file.

Usage (as a function):
    get_combination_list_with_accuracy(
        train_excel_file,
        test_excel_file,
        sfs_results_excel_file,
        output_excel_file='Step2_Results_v2.xlsx',
        group_column='Group',
        feature_range=(5, 20),
        c_set=None,
        gamma_set=None,
        n_splits=10,
        random_state=0,
        logger=None
    )
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import logging
import os

def get_sensitivity(y_true, y_pred):
    # Sensitivity = TP / (TP + FN)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TP = cm[1, 1]
        FN = cm[1, 0]
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return 0.0

def get_specificity(y_true, y_pred):
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        TN = cm[0, 0]
        FP = cm[0, 1]
        return TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return 0.0

def get_combination_list_with_accuracy(
    train_excel_file,
    test_excel_file,
    sfs_results_excel_file,
    output_excel_file='Step2_Results_v2.xlsx',
    group_column='Group',
    feature_range=(5, 20),
    c_set=None,
    gamma_set=None,
    n_splits=10,
    random_state=0,
    logger=None
):
    """
    For each sheet in the train and test Excel files, evaluates SVM models on feature subsets and hyperparameter combinations.
    Saves results to an output Excel file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    if c_set is None:
        c_set = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
    if gamma_set is None:
        gamma_set = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    feature_len_list = np.arange(feature_range[0], feature_range[1])

    logger.info(f"Loading train file: {train_excel_file}")
    logger.info(f"Loading test file: {test_excel_file}")
    logger.info(f"Loading SFS results file: {sfs_results_excel_file}")

    excel_train = pd.ExcelFile(train_excel_file)
    excel_test = pd.ExcelFile(test_excel_file)
    sfs_results = pd.ExcelFile(sfs_results_excel_file)
    step2_results = {}

    for sheet_name in excel_train.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")
        df_train = pd.read_excel(train_excel_file, sheet_name=sheet_name)
        df_test = pd.read_excel(test_excel_file, sheet_name=sheet_name)
        df_features = pd.read_excel(sfs_results_excel_file, sheet_name=sheet_name)
        
        
        
        
        
        #change it 
        # Encode group labels
        #df_train[group_column] = df_train[group_column].replace({'Control': 0, 'Flatfoot': 1})
        #df_test[group_column] = df_test[group_column].replace({'Control': 0, 'Flatfoot': 1})
        # Clean group labels
        df_train[group_column] = df_train[group_column].astype(str).str.strip()
        df_test[group_column] = df_test[group_column].astype(str).str.strip()

        # Encode group labels for Autism 2024
        label_map = {
            "Control": 0,
            "Autism": 1
        }

        df_train[group_column] = df_train[group_column].map(label_map)
        df_test[group_column] = df_test[group_column].map(label_map)

        # Check if any labels were not converted
        if df_train[group_column].isna().any() or df_test[group_column].isna().any():
            print("Train Group values after mapping:")
            print(df_train[group_column].value_counts(dropna=False))

            print("Test Group values after mapping:")
            print(df_test[group_column].value_counts(dropna=False))

            raise ValueError("Some Group labels were not mapped correctly. Check the exact group names in the Excel file.")

        # Convert to integer
        df_train[group_column] = df_train[group_column].astype(int)
        df_test[group_column] = df_test[group_column].astype(int)

        y_train = df_train[group_column].values
        y_test = df_test[group_column].values
############################
        # Feature segment (adjust as needed)
        segments_train = df_train.loc[:, 'Pelv_Angle_Y_MAX_SW':'Hip_Angle_Z_OHS']
        segments_test = df_test.loc[:, 'Pelv_Angle_Y_MAX_SW':'Hip_Angle_Z_OHS']

        skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        sc = StandardScaler()

        df_results = pd.DataFrame(
            columns=['CV_Accuracy', 'Test_Accuracy', 'Sensitivity', '#Features', 'C', 'gamma', 'CV_split', 'Random_State', 'Feature_idx', 'Confusion_Matrix', 'y_pred']
        )

        for s in range(len(df_features)):
            for feature_len in feature_len_list:
                res = np.array(literal_eval(str(df_features.iloc[:, -1][s])))
                sfs_feature_idx = res[:feature_len]
                X_train = segments_train.iloc[:, sfs_feature_idx].values
                X_train = sc.fit_transform(X_train)
                X_test = segments_test.iloc[:, sfs_feature_idx].values
                X_test = sc.transform(X_test)
                for C_val in c_set:
                    for gamma_val in gamma_set:
                        svm = SVC(kernel='rbf', C=C_val, gamma=gamma_val, verbose=False)
                        y_true_unflat_list, y_pred_unflat_list = [], []
                        for train_idx, test_idx in skf_cv.split(X_train, y_train):
                            x_tr, y_tr = X_train[train_idx], y_train[train_idx]
                            x_te, y_te = X_train[test_idx], y_train[test_idx]
                            svm.fit(x_tr, y_tr)
                            y_pred = svm.predict(x_te)
                            y_true_unflat_list.append(y_te[:])
                            y_pred_unflat_list.append(y_pred[:])
                        y_true_list = [item for sublist in y_true_unflat_list for item in sublist]
                        y_pred_list = [item for sublist in y_pred_unflat_list for item in sublist]
                        svm.fit(X_train, y_train)
                        y_pred_test = svm.predict(X_test)
                        sensitivity = get_sensitivity(y_test, y_pred_test)
                        specificity = get_specificity(y_test, y_pred_test)
                        cv_acc = accuracy_score(y_true_list, y_pred_list)
                        test_acc = accuracy_score(y_test, y_pred_test)
                        if sensitivity >= 0.5 and specificity > 0.5 and cv_acc >= 0.5:
                            df_results = pd.concat([
                                df_results,
                                pd.DataFrame([[cv_acc, test_acc, sensitivity, len(sfs_feature_idx), C_val, gamma_val, n_splits, random_state, s, confusion_matrix(y_test, y_pred_test), y_pred_test]],
                                             columns=['CV_Accuracy', 'Test_Accuracy', 'Sensitivity', '#Features', 'C', 'gamma', 'CV_split', 'Random_State', 'Feature_idx', 'Confusion_Matrix', 'y_pred'])
                            ], ignore_index=True)
                logger.info(f"Feature_idx: {s} #Features: {len(sfs_feature_idx)}")
        step2_results[sheet_name] = df_results

    logger.info(f"Saving results to {output_excel_file}")
    with pd.ExcelWriter(output_excel_file) as writer:
        for sheet_name, df_result in step2_results.items():
            df_result.to_excel(writer, sheet_name=sheet_name, index=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 2: Combinations - Evaluate SVM on feature subsets and hyperparameters.")
    parser.add_argument('--train', required=True, help='Path to train Excel file')
    parser.add_argument('--test', required=True, help='Path to test Excel file')
    parser.add_argument('--sfs', required=True, help='Path to SFS results Excel file (from Step 1)')
    parser.add_argument('--output', default='Step2_Results_v2.xlsx', help='Output Excel file')
    args = parser.parse_args()
    get_combination_list_with_accuracy(
        train_excel_file=args.train,
        test_excel_file=args.test,
        sfs_results_excel_file=args.sfs,
        output_excel_file=args.output
    ) 