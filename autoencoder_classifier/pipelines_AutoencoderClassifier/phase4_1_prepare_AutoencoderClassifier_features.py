import os
import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import f_classif

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

def clean_numeric_dataframe(df, group_column="Group", participant_column="Participant"):
    """
    Keep only numeric feature columns.
    Remove Group and Participant columns to avoid data leakage.
    """
    df = df.copy()

    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in dataframe.")

    y = df[group_column]

    columns_to_drop = [group_column]

    if participant_column in df.columns:
        columns_to_drop.append(participant_column)

    feature_df = df.drop(columns=columns_to_drop, errors="ignore")

    numeric_features = feature_df.select_dtypes(include=[np.number])
    numeric_features = numeric_features.dropna(axis=1, how="all")
    numeric_features = numeric_features.fillna(numeric_features.median())

    return numeric_features, y


def encode_labels(y):
    """
    Convert class labels to numeric values.
    Example:
    Control -> 0
    Flatfoot -> 1
    """
    y_encoded = y.copy()

    if y_encoded.dtype == "object":
        y_encoded = y_encoded.astype("category").cat.codes

    return y_encoded


def create_AutoencoderClassifier_feature_sets(
    train_file,
    output_file,
    feature_sizes=range(5, 20),
    group_column="Group",
    participant_column="Participant",
):
    """
    Create AutoencoderClassifier feature subsets using ANOVA F-score ranking.

    The test file is not used here to avoid data leakage.
    """
    logger.info(f"Reading train file: {train_file}")

    excel_file = pd.ExcelFile(train_file)
    all_results = []

    for sheet_name in excel_file.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(train_file, sheet_name=sheet_name)

        X, y = clean_numeric_dataframe(
            df,
            group_column=group_column,
            participant_column=participant_column,
        )

        y_encoded = encode_labels(y)

        if X.shape[1] == 0:
            logger.warning(f"No numeric features found in sheet: {sheet_name}")
            continue

        if len(np.unique(y_encoded)) < 2:
            logger.warning(
                f"Sheet {sheet_name} has fewer than two classes. Skipping."
            )
            continue

        f_scores, p_values = f_classif(X, y_encoded)

        ranking_df = pd.DataFrame({
            "Feature": X.columns,
            "Feature_Index": list(range(len(X.columns))),
            "F_Score": f_scores,
            "P_Value": p_values,
        })

        ranking_df = ranking_df.replace([np.inf, -np.inf], np.nan)
        ranking_df = ranking_df.dropna(subset=["F_Score"])

        ranking_df = ranking_df.sort_values(
            by="F_Score",
            ascending=False,
        ).reset_index(drop=True)

        for num_features in feature_sizes:
            if num_features > len(ranking_df):
                logger.warning(
                    f"Sheet {sheet_name}: requested {num_features} features, "
                    f"but only {len(ranking_df)} features are available."
                )
                continue

            selected = ranking_df.head(num_features)

            selected_indices = selected["Feature_Index"].astype(int).tolist()
            selected_features = selected["Feature"].astype(str).tolist()

            all_results.append({
                "Dataset_Sheet": sheet_name,
                "Model": "AutoencoderClassifier",
                "#Features": num_features,

                # Numeric feature indices
                "Features": str(selected_indices),

                # Real feature names used by Phase 4.2
                "Feature_Names": str(selected_features),

                "Ranking_Method": "ANOVA_F_Score",
                "Top_F_Score": float(selected["F_Score"].iloc[0]),
                "Mean_F_Score": float(selected["F_Score"].mean()),
            })

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        raise ValueError("No AutoencoderClassifier feature sets were created.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        results_df.to_excel(
            writer,
            sheet_name="AutoencoderClassifier_Feature_Sets",
            index=False,
        )

    logger.info(f"Saved AutoencoderClassifier Phase 4.1 results to: {output_file}")


# ============================================================
# Main
# ============================================================

def main():
    logger.info(">>>>>>>>>> Starting AutoencoderClassifier Phase 4.1: Preparing AutoencoderClassifier Feature Sets <<<<<<<<<<")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    group_column = config.get("SFS", {}).get("labels_column", "Group")
    participant_column = config.get("AutoencoderClassifier", {}).get("participant_column", "Participant")

    train_file = os.path.join(
        PROJECT_ROOT,
        config["split_datasets"]["output_train_file"],
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["sfs_results_excel_file_AutoencoderClassifier"],
    )

    create_AutoencoderClassifier_feature_sets(
        train_file=train_file,
        output_file=output_file,
        feature_sizes=range(5, 20),
        group_column=group_column,
        participant_column=participant_column,
    )

    logger.info(">>>>>>>>>> Completed AutoencoderClassifier Phase 4.1: Preparing AutoencoderClassifier Feature Sets <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in AutoencoderClassifier Phase 4.1: {e}")
        raise e
