"""
get_XGBoost_Scores.py

This script performs Phase 4.3: Scores XGBoost.

It is the XGBoost version of:
    src/gaitml/models/get_RBF_Scores.py

It reads:
    1) XGBoost SFS results from Phase 4.1
    2) XGBoost combination results from Phase 4.2

Then it calculates:
    Specificity
    Sensitivity
    NPV
    PPV
    Likelihood_Ratio
    F1
    MCC

And saves the final XGBoost scores file.
"""

import pandas as pd
import numpy as np
from ast import literal_eval
import logging

from ..utils.utils_metrics import (
    get_MCC_from_cm,
    get_specificity_from_cm,
    get_npv_from_cm,
    get_ppv_from_cm,
    get_f1_from_cm,
    get_likelihood_ratios_from_cm,
    parse_confusion_matrix
)


def get_XGBoost_scores(
    sfs_results_file,
    combination_results_file,
    output_file="Scores_XGBoost.xlsx",
    logger=None
):
    """
    Calculate comprehensive performance metrics for XGBoost models.

    Args:
        sfs_results_file:
            Path to XGBoost SFS results Excel file from Phase 4.1.

        combination_results_file:
            Path to XGBoost combination results Excel file from Phase 4.2.

        output_file:
            Output Excel file path.

        logger:
            Logger instance.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Loading XGBoost SFS results from: {sfs_results_file}")
    logger.info(f"Loading XGBoost combination results from: {combination_results_file}")

    excel_step1 = pd.ExcelFile(sfs_results_file)
    excel_step2 = pd.ExcelFile(combination_results_file)

    score_xgboost = {}

    for sheet_name in excel_step1.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        if sheet_name not in excel_step2.sheet_names:
            logger.warning(f"Sheet {sheet_name} not found in combination file. Skipping.")
            continue

        df_step1 = pd.read_excel(excel_step1, sheet_name=sheet_name)
        df_step2 = pd.read_excel(excel_step2, sheet_name=sheet_name)

        if df_step2.empty:
            logger.warning(f"No XGBoost combinations found for sheet: {sheet_name}")
            score_xgboost[sheet_name] = pd.DataFrame()
            continue

        required_columns = [
            "Feature_idx",
            "#Features",
            "Confusion_Matrix",
            "CV_Accuracy",
            "Test_Accuracy",
            "Sensitivity",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "CV_split",
            "Random_State"
        ]

        for col in required_columns:
            if col not in df_step2.columns:
                raise ValueError(
                    f"Column '{col}' not found in sheet '{sheet_name}' "
                    f"of {combination_results_file}"
                )

        df_results = pd.DataFrame(
            columns=[
                "Name_Model",
                "#Features",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "CV_split",
                "Random_State",
                "Feature_idx",
                "Model_Type",
                "CV_accuracy",
                "Test_Accuracy",
                "Specificity",
                "Sensitivity",
                "NPV",
                "PPV",
                "Likelihood_Ratio",
                "F1",
                "MCC",
                "Confusion_Matrix",
                "Features"
            ]
        )

        for counter in range(len(df_step2)):

            feature_idx = int(df_step2.loc[counter, "Feature_idx"])
            feature_count = int(df_step2.loc[counter, "#Features"])

            # Get feature list from SFS Phase 4.1 result
            features = np.array(
                literal_eval(str(df_step1.iloc[:, -1][feature_idx]))
            )

            feat_comb = features[:feature_count]

            # Parse confusion matrix from Phase 4.2
            matrix = parse_confusion_matrix(
                str(df_step2.loc[counter, "Confusion_Matrix"])
            )

            if matrix.shape != (2, 2):
                logger.warning(
                    f"Confusion matrix is not 2x2 for sheet {sheet_name}, "
                    f"row {counter}. Skipping."
                )
                continue

            tn, fp, fn, tp = matrix.ravel()

            specificity_val = get_specificity_from_cm(tn, fp, fn, tp)
            npv_val = get_npv_from_cm(tn, fn)
            ppv_val = get_ppv_from_cm(tp, fp)
            mcc_val = get_MCC_from_cm(tn, fp, fn, tp)
            f1_val = get_f1_from_cm(tp, fp, fn, tn)
            likelihood_ratio_val = get_likelihood_ratios_from_cm(tp, fp, fn, tn)

            # If Step 2 file has an index column from Excel, use it as Name_Model.
            # Otherwise, create a clean XGBoost model name.
            try:
                name_model = df_step2.iloc[counter, 0]
            except Exception:
                name_model = (
                    f"XGBoost_NE{df_step2.loc[counter, 'n_estimators']}_"
                    f"MD{df_step2.loc[counter, 'max_depth']}_"
                    f"LR{df_step2.loc[counter, 'learning_rate']}_"
                    f"F{feature_count}"
                )

            new_row = pd.DataFrame({
                "Name_Model": [name_model],
                "#Features": [feature_count],
                "n_estimators": [df_step2.loc[counter, "n_estimators"]],
                "max_depth": [df_step2.loc[counter, "max_depth"]],
                "learning_rate": [df_step2.loc[counter, "learning_rate"]],
                "subsample": [df_step2.loc[counter, "subsample"]],
                "colsample_bytree": [df_step2.loc[counter, "colsample_bytree"]],
                "CV_split": [df_step2.loc[counter, "CV_split"]],
                "Random_State": [df_step2.loc[counter, "Random_State"]],
                "Feature_idx": [feature_idx],
                "Model_Type": ["XGBoost"],
                "CV_accuracy": [df_step2.loc[counter, "CV_Accuracy"]],
                "Test_Accuracy": [df_step2.loc[counter, "Test_Accuracy"]],
                "Specificity": [specificity_val],
                "Sensitivity": [df_step2.loc[counter, "Sensitivity"]],
                "NPV": [npv_val],
                "PPV": [ppv_val],
                "Likelihood_Ratio": [likelihood_ratio_val],
                "F1": [f1_val],
                "MCC": [mcc_val],
                "Confusion_Matrix": [df_step2.loc[counter, "Confusion_Matrix"]],
                "Features": [str(feat_comb)]
            })

            df_results = pd.concat(
                [df_results, new_row],
                ignore_index=True
            )

        score_xgboost[sheet_name] = df_results

    logger.info(f"Saving XGBoost scores to: {output_file}")

    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df_results in score_xgboost.items():
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info("XGBoost scores calculation completed successfully.")

    return score_xgboost


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4.3: Scores XGBoost - Calculate metrics for XGBoost models."
    )

    parser.add_argument(
        "--sfs",
        required=True,
        help="Path to XGBoost SFS results Excel file from Phase 4.1"
    )

    parser.add_argument(
        "--combinations",
        required=True,
        help="Path to XGBoost combination results Excel file from Phase 4.2"
    )

    parser.add_argument(
        "--output",
        default="Scores_XGBoost.xlsx",
        help="Output Excel file path"
    )

    args = parser.parse_args()

    get_XGBoost_scores(
        sfs_results_file=args.sfs,
        combination_results_file=args.combinations,
        output_file=args.output
    )