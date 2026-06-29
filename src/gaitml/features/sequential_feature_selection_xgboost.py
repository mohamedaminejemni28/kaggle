"""
sequential_feature_selection_xgboost.py

XGBoost version of Sequential Feature Selection.
This file is used by:
    pipelines_xgboost/phase4_1_SFS_XGBoost.py
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier

from gaitml import logger


def sequential_feature_selection_xgboost(
    input_excel_file,
    output_excel_file,
    group_column="Group",
    classes_dict=None,
    n_features=50,
    n_estimators_set=None,
    max_depth_set=None,
    learning_rate_set=None,
    kfold_n_splits=10,
    random_state=0
):
    if n_estimators_set is None:
        n_estimators_set = [50, 100]

    if max_depth_set is None:
        max_depth_set = [2, 3]

    if learning_rate_set is None:
        learning_rate_set = [0.05, 0.1]

    logger.info(f"Loading training Excel file: {input_excel_file}")

    excel_file = pd.ExcelFile(input_excel_file)
    results_per_sheet = {}

    for sheet_name in excel_file.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(input_excel_file, sheet_name=sheet_name)

        if group_column not in df.columns:
            raise ValueError(
                f"Column '{group_column}' not found in sheet '{sheet_name}'"
            )

        # Clean labels
        df[group_column] = df[group_column].astype(str).str.strip()

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[group_column])

        # Same feature segment as original SVM pipeline
        X = df.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"].values

        skf_cv = StratifiedKFold(
            n_splits=kfold_n_splits,
            shuffle=True,
            random_state=random_state
        )

        df_results = pd.DataFrame(
            columns=[
                "CV_Accuracy",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "#Features",
                "Ordered Indices"
            ]
        )

        for n_estimators in n_estimators_set:
            for max_depth in max_depth_set:
                for learning_rate in learning_rate_set:

                    logger.info(
                        f"Running XGBoost SFS | "
                        f"n_estimators={n_estimators}, "
                        f"max_depth={max_depth}, "
                        f"learning_rate={learning_rate}"
                    )

                    model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1
                    )

                    sfs = SFS(
                        estimator=model,
                        k_features=n_features,
                        forward=True,
                        floating=False,
                        verbose=0,
                        scoring="accuracy",
                        cv=skf_cv,
                        n_jobs=-1
                    )

                    sfs.fit(X, y)

                    sfs_results = sfs.get_metric_dict()

                    ordered_indices = []
                    previous_features = set()

                    for i in range(1, n_features + 1):
                        current_features = set(sfs_results[i]["feature_idx"])
                        new_feature = current_features.difference(previous_features)

                        if len(new_feature) > 0:
                            ordered_indices.append(int(list(new_feature)[0]))

                        previous_features = current_features

                    X_selected = sfs.transform(X)

                    y_true_all = []
                    y_pred_all = []

                    for train_idx, test_idx in skf_cv.split(X_selected, y):
                        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        y_true_all.extend(y_test)
                        y_pred_all.extend(y_pred)

                    cv_accuracy = accuracy_score(y_true_all, y_pred_all)

                    new_row = pd.DataFrame(
                        [[
                            cv_accuracy,
                            n_estimators,
                            max_depth,
                            learning_rate,
                            len(ordered_indices),
                            ordered_indices
                        ]],
                        columns=[
                            "CV_Accuracy",
                            "n_estimators",
                            "max_depth",
                            "learning_rate",
                            "#Features",
                            "Ordered Indices"
                        ]
                    )

                    df_results = pd.concat(
                        [df_results, new_row],
                        ignore_index=True
                    )

                    logger.info(
                        f"Sheet={sheet_name} | "
                        f"CV_Accuracy={cv_accuracy} | "
                        f"#Features={len(ordered_indices)}"
                    )

        results_per_sheet[sheet_name] = df_results

    logger.info(f"Saving XGBoost SFS results to: {output_excel_file}")

    with pd.ExcelWriter(output_excel_file) as writer:
        for sheet_name, df_result in results_per_sheet.items():
            df_result.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("XGBoost SFS completed successfully.")