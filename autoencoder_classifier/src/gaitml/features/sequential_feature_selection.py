"""
Sequential Feature Selection (SFS) with SVM (RBF) for Each Sheet in Excel

This script performs SFS for each sheet in an Excel file and saves the results.
"""

"""
TO-DO:

    The features (independent variables) used for Cross validation will differ from dataset-to-dataset
        Make them dynamic
        X = df.loc[:, 'Pelv_Angle_Y_MAX_SW':'Hip_Angle_Z_OHS'].values

    Instead of using for loops for C and Gamma, there is a way 
        that returns these combinations one after the another.

    Add logger for logging the hyperparameters that are used in hyper-parameter optimization
    Add logger for k-folds done, and other parameters used for SFS.

    We can add Precision and recall (these are good for medical data evaluations)
        Additional to "CV_accuracy" metric
        acc = accuracy_score(y_true_list, y_pred_list)

    Different classes or functions for applying SFS to different ML algorithms. 
        This code does it for SVM RBF, but we can add other algorithms like LR, DT, RF, NN, etc.
    
    Save the Standard Scaler so that it can be used in transforming the test set.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from gaitml import logger

def sequential_feature_selection(
    input_excel_file: str,
    output_excel_file: str,
    group_column: str,
    classes_dict: dict,
    n_features: int,
    c_set=None,
    gamma_set=None,
    kfold_n_splits: int = 10,
    random_state: int = 0
):
    """
    Perform SFS with SVM (RBF) for each sheet in the Excel file.

    Args:
        input_excel_file (str): Path to the input Excel file with multiple sheets.
        output_excel_file (str): Output Excel file for saving SFS results.
        group_column (str): Name of the target column.
        classes_dict (dict): Dictionary for label encoding classes (target variables)
        n_features (int): Number of features to select.
        c_set (list): List of C values for SVM.
        gamma_set (list): List of gamma values for SVM.
        kfold_n_splits (int): Number of folds for StratifiedKFold.
        random_state (int): Random seed for reproducibility.
    """
    if c_set == []:
        c_set = [0.1, 1, 25, 50, 75, 100, 125, 150]
    if gamma_set == []:
        gamma_set = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    logger.info(f"Loading Excel file: {input_excel_file}")
    excel_in = pd.ExcelFile(input_excel_file)
    results_per_sheet = {}

    for sheet_name in excel_in.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")
        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)
        if group_column not in df.columns:
            logger.error(f"Column '{group_column}' not found in {sheet_name}.")

        # Encode target variable
        df[group_column] = df[group_column].replace(classes_dict) # classes_dict = {'Control': 0, 'Flatfoot': 1}
        y = df[group_column].values

        # Define feature segment (customize as needed)
        X = df.loc[:, 'Pelv_Angle_Y_MAX_SW':'Hip_Angle_Z_OHS'].values

        sc = StandardScaler()
        X = sc.fit_transform(X)

        kfold_cv = StratifiedKFold(n_splits=kfold_n_splits, shuffle=True, random_state=random_state)
        df_results = pd.DataFrame(columns=['CV_Accuracy', 'C', 'Gamma', '#Features', 'Ordered Indices'])

        for C_val in c_set:
            for gamma_val in gamma_set:
                svm = SVC(kernel='rbf', C=C_val, gamma=gamma_val, verbose=False)
                sfs = SFS(
                    estimator=svm,
                    k_features=n_features,
                    forward=True,
                    floating=False,
                    verbose=0,
                    scoring='accuracy',
                    cv=kfold_cv,
                    n_jobs=-1
                )
                sfs.fit(X, y)
                sfs_results = sfs.get_metric_dict()

                # Get the order in which features were selected
                selected_features_order = []
                setA = set()
                for i in range(1, n_features + 1):
                    setB = setA
                    setA = set(sfs_results[i]['feature_names'])
                    newfeat = setA.difference(setB)
                    selected_features_order.append(int(newfeat.pop()))
                
                # Evaluate with cross-validation
                X_sfs = sfs.transform(X)
                y_true_list, y_pred_list = [], []
                for train_idx, test_idx in kfold_cv.split(X_sfs, y):
                    x_train, y_train = X_sfs[train_idx], y[train_idx]
                    x_test, y_test = X_sfs[test_idx], y[test_idx]
                    svm.fit(x_train, y_train)
                    y_pred = svm.predict(x_test)
                    y_true_list.extend(y_test)
                    y_pred_list.extend(y_pred)
                acc = accuracy_score(y_true_list, y_pred_list)
                logger.info(f"SFS with C: {C_val} and Gamma: {gamma_val} - Accuracy: {acc}")

                df_results = pd.concat([df_results, pd.DataFrame(
                    [[acc, C_val, gamma_val, len(sfs.k_feature_idx_), selected_features_order]],
                    columns=['CV_Accuracy', 'C', 'Gamma', '#Features', 'Ordered Indices']
                )], ignore_index=True)

        results_per_sheet[sheet_name] = df_results

    logger.info(f"Saving SFS results to {output_excel_file}")
    with pd.ExcelWriter(output_excel_file) as writer:
        for sheet_name, df_filtered in results_per_sheet.items():
            df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)
    logger.info("Completed execting sequential feature selection.")

if __name__ == "__main__":

    main_dir_path = "F:/python_venv/Gait-ML-Shap/"

    dataset_train_file = main_dir_path + "data/processed/ALL FEATURES OLDER - Pes Planus and Control NH Correlated features JUNE 25 Train.xlsx"
    output_SFS_scores_file = main_dir_path + "src/gaitml/features/phase4_1_SFS_results.xlsx"
    
    labels_column = "Group"
    classes_dict = {0: 'Control', 1: 'Flatfoot'}

    sequential_feature_selection(
        input_excel_file = dataset_train_file,
        output_excel_file = output_SFS_scores_file,
        group_column = labels_column, # Name of the target column.
        classes_dict = classes_dict, # Dictionary for label encoding classes (target variables)
        n_features = 50, #  Number of features to select.
        c_set = [1, 10], # List of C values for SVM.
        gamma_set = [0.1, 1, 10], # List of gamma values for SVM.
        kfold_n_splits = 10, # Number of folds for StratifiedKFold.
        random_state = 0
    )