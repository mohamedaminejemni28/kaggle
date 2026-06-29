import os
import sys
import json
import ast

import numpy as np
import pandas as pd


# ============================================================
# Add src directory to Python path
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from gaitml import logger
from gaitml.utils.helpers import read_yaml


# ============================================================
# Helpers
# ============================================================

def parse_list(value):
    """
    Convert Excel string lists back to Python lists.
    """
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [value]

    return [value]


def safe_json_load(value):
    """
    Read JSON config from Excel safely.
    """
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}

    return {}


def ensure_numeric_columns(df, columns):
    """
    Convert metric columns to numeric values.
    """
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def build_sort_columns(df):
    """
    Choose sorting columns that exist in the AutoencoderClassifier result file.
    """
    preferred_sort = [
        ("CV_Balanced_Accuracy_Mean", False),
        ("CV_F1_Mean", False),
        ("CV_ROC_AUC_Mean", False),
        ("CV_Accuracy_Mean", False),
        ("CV_Sensitivity_Mean", False),
        ("CV_Specificity_Mean", False),
        ("CV_Balanced_Accuracy_Std", True),
        ("#Features", True),
    ]

    sort_columns = []
    ascending = []

    for column, asc in preferred_sort:
        if column in df.columns:
            sort_columns.append(column)
            ascending.append(asc)

    if not sort_columns:
        raise ValueError("No valid CV metric columns found for sorting.")

    return sort_columns, ascending


def add_readable_model_name(df):
    """
    Add a readable Name_Model column if it does not already exist.
    """
    df = df.copy()

    if "Name_Model" not in df.columns:
        names = []

        for _, row in df.iterrows():
            candidate_id = row.get("Candidate_ID", "")
            num_features = row.get("#Features", "")
            config = safe_json_load(row.get("Model_Config", "{}"))

            encoder_layers = config.get("encoder_layers", "NA")
            latent_dim = config.get("latent_dim", "NA")
            reconstruction_weight = config.get("reconstruction_weight", "NA")
            lr = config.get("learning_rate", "NA")

            names.append(
                f"AutoencoderClassifier_Candidate_{candidate_id}_"
                f"{num_features}F_"
                f"ENC{encoder_layers}_"
                f"LAT{latent_dim}_"
                f"RW{reconstruction_weight}_"
                f"LR{lr}"
            )

        df.insert(0, "Name_Model", names)

    return df


def clean_feature_columns(df):
    """
    Add a readable Feature_Count_Check and Feature_List_Length.
    """
    df = df.copy()

    if "Feature_Names" in df.columns:
        feature_lengths = []

        for value in df["Feature_Names"]:
            feature_lengths.append(len(parse_list(value)))

        df["Feature_List_Length"] = feature_lengths

    return df


def select_top3_by_feature_count(df, feature_counts=None, top_n=3):
    """
    Select Top N models for each requested feature count.
    """
    if feature_counts is None:
        feature_counts = sorted(df["#Features"].dropna().astype(int).unique())

    selected_parts = []

    for count in feature_counts:
        subset = df[df["#Features"].astype(int) == int(count)].copy()

        if subset.empty:
            logger.warning(f"No AutoencoderClassifier results found for {count} features.")
            continue

        subset = subset.head(top_n).copy()
        subset["Top_Group"] = f"Top {top_n} for {count} features"

        selected_parts.append(subset)

    if not selected_parts:
        return pd.DataFrame()

    return pd.concat(selected_parts, ignore_index=True)


def save_excel_outputs(
    ranked_df,
    top3_overall_df,
    top3_by_features_df,
    best_per_feature_df,
    scores_file,
    top3_file,
):
    """
    Save ranked scores and Top 3 outputs.
    """
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    os.makedirs(os.path.dirname(top3_file), exist_ok=True)

    with pd.ExcelWriter(scores_file, engine="openpyxl") as writer:
        ranked_df.to_excel(writer, sheet_name="Ranked_All_AutoencoderClassifier", index=False)
        best_per_feature_df.to_excel(writer, sheet_name="Best_Per_Feature_Count", index=False)
        top3_overall_df.to_excel(writer, sheet_name="Top3_Overall", index=False)
        top3_by_features_df.to_excel(writer, sheet_name="Top3_By_Feature_Count", index=False)

    with pd.ExcelWriter(top3_file, engine="openpyxl") as writer:
        top3_overall_df.to_excel(writer, sheet_name="Top3_Overall", index=False)
        top3_by_features_df.to_excel(writer, sheet_name="Top3_By_Feature_Count", index=False)

    logger.info(f"Saved ranked AutoencoderClassifier scores to: {scores_file}")
    logger.info(f"Saved Top 3 AutoencoderClassifier scores to: {top3_file}")


# ============================================================
# Main
# ============================================================

def main():
    logger.info(">>>>>>>>>> Starting AutoencoderClassifier Phase 4.3: Selecting Best AutoencoderClassifier Scores <<<<<<<<<<")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    input_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["combination_results_file_AutoencoderClassifier"],
    )

    scores_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_AutoencoderClassifier"],
    )

    top3_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_top3_AutoencoderClassifier"],
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"AutoencoderClassifier Phase 4.2 result file not found: {input_file}")

    logger.info(f"Reading AutoencoderClassifier Phase 4.2 results from: {input_file}")

    df = pd.read_excel(input_file)

    if df.empty:
        raise ValueError("AutoencoderClassifier Phase 4.2 results file is empty.")

    required_columns = {
        "Candidate_ID",
        "#Features",
        "Feature_Names",
        "Model_Config",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in AutoencoderClassifier results: {sorted(missing_columns)}")

    metric_columns = [
        "CV_Accuracy_Mean",
        "CV_Accuracy_Std",
        "CV_Balanced_Accuracy_Mean",
        "CV_Balanced_Accuracy_Std",
        "CV_Precision_Mean",
        "CV_Precision_Std",
        "CV_Sensitivity_Mean",
        "CV_Sensitivity_Std",
        "CV_Specificity_Mean",
        "CV_Specificity_Std",
        "CV_F1_Mean",
        "CV_F1_Std",
        "CV_ROC_AUC_Mean",
        "CV_ROC_AUC_Std",
        "#Features",
        "Best_Epoch_Median",
    ]

    df = ensure_numeric_columns(df, metric_columns)
    df = add_readable_model_name(df)
    df = clean_feature_columns(df)

    sort_columns, ascending = build_sort_columns(df)

    logger.info(f"Sorting AutoencoderClassifier models by: {sort_columns}")

    ranked_df = df.sort_values(
        by=sort_columns,
        ascending=ascending,
        na_position="last",
    ).reset_index(drop=True)

    ranked_df.insert(0, "Overall_Rank", range(1, len(ranked_df) + 1))

    # Top 3 overall
    top3_overall_df = ranked_df.head(3).copy()
    top3_overall_df["Top_Group"] = "Top 3 overall"

    # Top 3 for selected feature counts
    AutoencoderClassifier_config = config.get("AutoencoderClassifier", {})

    requested_feature_counts = AutoencoderClassifier_config.get("top3_feature_counts", [5, 6, 10])

    top3_by_features_df = select_top3_by_feature_count(
        ranked_df,
        feature_counts=requested_feature_counts,
        top_n=3,
    )

    # Best model for each available feature count
    best_per_feature_df = (
        ranked_df
        .sort_values(by=sort_columns, ascending=ascending, na_position="last")
        .groupby("#Features", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    save_excel_outputs(
        ranked_df=ranked_df,
        top3_overall_df=top3_overall_df,
        top3_by_features_df=top3_by_features_df,
        best_per_feature_df=best_per_feature_df,
        scores_file=scores_file,
        top3_file=top3_file,
    )

    logger.info("Top 3 AutoencoderClassifier overall models:")
    for _, row in top3_overall_df.iterrows():
        logger.info(
            f"Rank {int(row['Overall_Rank'])} | "
            f"Candidate {row['Candidate_ID']} | "
            f"{int(row['#Features'])} features | "
            f"CV Balanced Accuracy = {row.get('CV_Balanced_Accuracy_Mean', np.nan):.3f} | "
            f"CV F1 = {row.get('CV_F1_Mean', np.nan):.3f}"
        )

    logger.info(">>>>>>>>>> Completed AutoencoderClassifier Phase 4.3: Selecting Best AutoencoderClassifier Scores <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in AutoencoderClassifier Phase 4.3: {e}")
        raise e
