"""
get_model_combinations_with_accuracy_xgboost.py

This script performs Step 2: Combinations using XGBoost instead of SVM.

It is the XGBoost version of:
    src/gaitml/models/get_model_combinations_with_accuracy.py

It does NOT modify the old SVM file.
It creates similar output but with XGBoost hyperparameters:

    n_estimators
    max_depth
    learning_rate
    subsample
    colsample_bytree

Output columns include:
    CV_Accuracy
    Test_Accuracy
    Sensitivity
    Specificity
    #Features
    n_estimators
    max_depth
    learning_rate
    subsample
    colsample_bytree
    CV_split
    Random_State
    Feature_idx
    Confusion_Matrix
    y_pred
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from ast import literal_eval
import logging


def get_sensitivity(y_true, y_pred):
    """
    Sensitivity = TP / (TP + FN)
    Also called Recall for the positive class.
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        TP = cm[1, 1]
        FN = cm[1, 0]
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    return 0.0


def get_specificity(y_true, y_pred):
    """
    Specificity = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        TN = cm[0, 0]
        FP = cm[0, 1]
        return TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return 0.0


def clean_and_encode_labels(df_train, df_test, group_column):
    """
    Clean and encode labels.

    This version is safer than hard-coding:
        Control -> 0
        Autism -> 1

    It can work with:
        Control / Autism
        Control / Flatfoot
        Young / Older
        0 / 1

    The encoder learns labels from train + test together.
    """

    df_train[group_column] = df_train[group_column].astype(str).str.strip()
    df_test[group_column] = df_test[group_column].astype(str).str.strip()

    label_encoder = LabelEncoder()

    all_labels = pd.concat(
        [df_train[group_column], df_test[group_column]],
        axis=0
    )

    label_encoder.fit(all_labels)

    df_train[group_column] = label_encoder.transform(df_train[group_column])
    df_test[group_column] = label_encoder.transform(df_test[group_column])

    return df_train, df_test, label_encoder


def get_combination_list_with_accuracy_xgboost(
    train_excel_file,
    test_excel_file,
    sfs_results_excel_file,
    output_excel_file="Step2_Results_XGBoost.xlsx",
    group_column="Group",
    feature_range=(5, 20),
    n_estimators_set=None,
    max_depth_set=None,
    learning_rate_set=None,
    subsample_set=None,
    colsample_bytree_set=None,
    n_splits=10,
    random_state=0,
    logger=None
):
    """
    For each sheet in the train and test Excel files,
    evaluates XGBoost models on feature subsets and hyperparameter combinations.

    Parameters:
        train_excel_file:
            Training Excel file.

        test_excel_file:
            Testing Excel file.

        sfs_results_excel_file:
            SFS result file from Step 1.

        output_excel_file:
            Output Excel file for Step 2 XGBoost results.

        group_column:
            Name of the target column.

        feature_range:
            Range of selected features.
            Example: (5, 20) means 5 to 19 features.

        n_estimators_set:
            List of XGBoost n_estimators values.

        max_depth_set:
            List of XGBoost max_depth values.

        learning_rate_set:
            List of XGBoost learning_rate values.

        subsample_set:
            List of XGBoost subsample values.

        colsample_bytree_set:
            List of XGBoost colsample_bytree values.

        n_splits:
            Number of cross-validation folds.

        random_state:
            Random seed.

        logger:
            Logger object.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    # XGBoost hyperparameters
    if n_estimators_set is None:
        n_estimators_set = [50, 100, 150]

    if max_depth_set is None:
        max_depth_set = [2, 3, 4]

    if learning_rate_set is None:
        learning_rate_set = [0.01, 0.05, 0.1]

    if subsample_set is None:
        subsample_set = [0.8, 1.0]

    if colsample_bytree_set is None:
        colsample_bytree_set = [0.8, 1.0]

    feature_len_list = np.arange(feature_range[0], feature_range[1])

    logger.info(f"Loading train file: {train_excel_file}")
    logger.info(f"Loading test file: {test_excel_file}")
    logger.info(f"Loading XGBoost SFS results file: {sfs_results_excel_file}")

    excel_train = pd.ExcelFile(train_excel_file)
    excel_test = pd.ExcelFile(test_excel_file)
    sfs_results = pd.ExcelFile(sfs_results_excel_file)

    step2_results = {}

    for sheet_name in excel_train.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        df_train = pd.read_excel(train_excel_file, sheet_name=sheet_name)
        df_test = pd.read_excel(test_excel_file, sheet_name=sheet_name)
        df_features = pd.read_excel(sfs_results_excel_file, sheet_name=sheet_name)

        if sheet_name not in excel_test.sheet_names:
            logger.warning(f"Sheet {sheet_name} not found in test file. Skipping.")
            continue

        if sheet_name not in sfs_results.sheet_names:
            logger.warning(f"Sheet {sheet_name} not found in SFS file. Skipping.")
            continue

        if group_column not in df_train.columns:
            raise ValueError(f"Column '{group_column}' not found in train sheet: {sheet_name}")

        if group_column not in df_test.columns:
            raise ValueError(f"Column '{group_column}' not found in test sheet: {sheet_name}")

        # Encode group labels
        df_train, df_test, label_encoder = clean_and_encode_labels(
            df_train=df_train,
            df_test=df_test,
            group_column=group_column
        )

        y_train = df_train[group_column].values
        y_test = df_test[group_column].values

        # Feature segment: same as SVM file
        segments_train = df_train.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"]
        segments_test = df_test.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"]

        skf_cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        df_results = pd.DataFrame(
            columns=[
                "CV_Accuracy",
                "Test_Accuracy",
                "Sensitivity",
                "Specificity",
                "#Features",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "CV_split",
                "Random_State",
                "Feature_idx",
                "Confusion_Matrix",
                "y_pred"
            ]
        )

        for s in range(len(df_features)):

            for feature_len in feature_len_list:

                try:
                    res = np.array(literal_eval(str(df_features.iloc[:, -1][s])))
                except Exception:
                    logger.warning(f"Could not read selected features at row {s} in sheet {sheet_name}")
                    continue

                sfs_feature_idx = res[:feature_len]

                if len(sfs_feature_idx) == 0:
                    continue

                X_train = segments_train.iloc[:, sfs_feature_idx].values
                X_test = segments_test.iloc[:, sfs_feature_idx].values

                for n_estimators in n_estimators_set:
                    for max_depth in max_depth_set:
                        for learning_rate in learning_rate_set:
                            for subsample in subsample_set:
                                for colsample_bytree in colsample_bytree_set:

                                    xgb_model = XGBClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        subsample=subsample,
                                        colsample_bytree=colsample_bytree,
                                        objective="binary:logistic",
                                        eval_metric="logloss",
                                        random_state=random_state,
                                        n_jobs=-1
                                    )

                                    y_true_unflat_list = []
                                    y_pred_unflat_list = []

                                    for train_idx, test_idx in skf_cv.split(X_train, y_train):
                                        x_tr = X_train[train_idx]
                                        y_tr = y_train[train_idx]

                                        x_te = X_train[test_idx]
                                        y_te = y_train[test_idx]

                                        xgb_model.fit(x_tr, y_tr)
                                        y_pred_cv = xgb_model.predict(x_te)

                                        y_true_unflat_list.append(y_te[:])
                                        y_pred_unflat_list.append(y_pred_cv[:])

                                    y_true_list = [
                                        item
                                        for sublist in y_true_unflat_list
                                        for item in sublist
                                    ]

                                    y_pred_list = [
                                        item
                                        for sublist in y_pred_unflat_list
                                        for item in sublist
                                    ]

                                    # Train final model on all train data
                                    xgb_model.fit(X_train, y_train)

                                    # Test prediction
                                    y_pred_test = xgb_model.predict(X_test)

                                    sensitivity = get_sensitivity(y_test, y_pred_test)
                                    specificity = get_specificity(y_test, y_pred_test)

                                    cv_acc = accuracy_score(y_true_list, y_pred_list)
                                    test_acc = accuracy_score(y_test, y_pred_test)

                                    # Same filtering idea as SVM file
                                    if sensitivity >= 0.5 and specificity > 0.5 and cv_acc >= 0.5:

                                        new_row = pd.DataFrame(
                                            [[
                                                cv_acc,
                                                test_acc,
                                                sensitivity,
                                                specificity,
                                                len(sfs_feature_idx),
                                                n_estimators,
                                                max_depth,
                                                learning_rate,
                                                subsample,
                                                colsample_bytree,
                                                n_splits,
                                                random_state,
                                                s,
                                                confusion_matrix(y_test, y_pred_test),
                                                y_pred_test
                                            ]],
                                            columns=[
                                                "CV_Accuracy",
                                                "Test_Accuracy",
                                                "Sensitivity",
                                                "Specificity",
                                                "#Features",
                                                "n_estimators",
                                                "max_depth",
                                                "learning_rate",
                                                "subsample",
                                                "colsample_bytree",
                                                "CV_split",
                                                "Random_State",
                                                "Feature_idx",
                                                "Confusion_Matrix",
                                                "y_pred"
                                            ]
                                        )

                                        df_results = pd.concat(
                                            [df_results, new_row],
                                            ignore_index=True
                                        )

                logger.info(
                    f"Sheet: {sheet_name} | "
                    f"Feature_idx: {s} | "
                    f"#Features: {len(sfs_feature_idx)}"
                )

        step2_results[sheet_name] = df_results

    logger.info(f"Saving XGBoost Step 2 results to {output_excel_file}")

    with pd.ExcelWriter(output_excel_file) as writer:
        for sheet_name, df_result in step2_results.items():
            df_result.to_excel(writer, sheet_name=sheet_name, index=True)

    logger.info("XGBoost Step 2 completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 2: Combinations - Evaluate XGBoost on feature subsets and hyperparameters."
    )

    parser.add_argument(
        "--train",
        required=True,
        help="Path to train Excel file"
    )

    parser.add_argument(
        "--test",
        required=True,
        help="Path to test Excel file"
    )

    parser.add_argument(
        "--sfs",
        required=True,
        help="Path to XGBoost SFS results Excel file from Step 1"
    )

    parser.add_argument(
        "--output",
        default="Step2_Results_XGBoost.xlsx",
        help="Output Excel file"
    )

    args = parser.parse_args()

    get_combination_list_with_accuracy_xgboost(
        train_excel_file=args.train,
        test_excel_file=args.test,
        sfs_results_excel_file=args.sfs,
        output_excel_file=args.output
    )