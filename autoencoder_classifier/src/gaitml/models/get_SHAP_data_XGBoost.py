"""
XGBoost SHAP Data Analysis Utilities

This module performs SHAP analysis on XGBoost models.

Inputs:
- XGBoost Top3 scores file
- Train dataset
- Test dataset

Outputs:
- Excel file with sample-level SHAP values
- Excel sheets with global SHAP feature ranking
- PNG SHAP summary images for each model
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier

from gaitml import logger


# ============================================================
# HELPERS
# ============================================================

def parse_selected_indices(value):
    """
    Convert selected feature indices from Excel text to Python list.

    Supported formats:
    "[1, 5, 10]"
    "[1 5 10]"
    "1 5 10"
    "1, 5, 10"
    """

    if isinstance(value, list):
        return [int(v) for v in value]

    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple)):
            return [int(v) for v in parsed]

        if isinstance(parsed, (int, float)):
            return [int(parsed)]

    except Exception:
        pass

    text = text.replace("[", "").replace("]", "").replace(",", " ")
    parts = text.split()

    try:
        return [int(float(p)) for p in parts]
    except Exception:
        return []


def get_feature_index_column(df_scores):
    """
    Find the column that contains the selected feature indices.
    """

    possible_columns = [
        "Features",
        "Selected Indices",
        "Feature_idx"
    ]

    for col in possible_columns:
        if col in df_scores.columns:
            return col

    raise ValueError(
        "No selected feature index column found. Expected one of: "
        f"{possible_columns}. Available columns are: {df_scores.columns.tolist()}"
    )


def get_cv_column(df_scores):
    """
    Handle different CV accuracy column names.
    """

    if "CV_Accuracy" in df_scores.columns:
        return "CV_Accuracy"

    if "CV_accuracy" in df_scores.columns:
        return "CV_accuracy"

    raise ValueError(
        "No CV accuracy column found. Expected 'CV_Accuracy' or 'CV_accuracy'. "
        f"Available columns are: {df_scores.columns.tolist()}"
    )


def get_feature_names(df):
    """
    Get biomechanical feature columns used in the pipeline.
    """

    try:
        return df.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"].columns.tolist()
    except Exception as e:
        raise ValueError(
            "Could not extract feature columns from Pelv_Angle_Y_MAX_SW to Hip_Angle_Z_OHS. "
            "Check that these columns exist in the train/test files."
        ) from e


def prepare_labels(df, labels_column):
    """
    Encode labels using the same mapping as the SVM pipeline.

    Young vs Older:
        Young = 0
        Older = 1

    Autism:
        Control = 0
        Autism = 1
    """

    if labels_column not in df.columns:
        raise ValueError(f"Column '{labels_column}' not found in dataset.")

    labels = df[labels_column].astype(str).str.strip()

    label_mapping = {
        "Young": 0,
        "Older": 1,
        "Control": 0,
        "Autism": 1,
        "0": 0,
        "1": 1,
        0: 0,
        1: 1,
        "Flatfoot": 1,
        "control": 0,
    }

    y = labels.replace(label_mapping)
    y = pd.to_numeric(y, errors="coerce")

    if y.isna().any():
        bad_labels = labels[y.isna()].unique()
        raise ValueError(
            f"Some labels could not be converted to 0/1: {bad_labels}. "
            f"Available labels are: {labels.unique()}"
        )

    y = y.astype(int).values

    logger.info("Label mapping used:")
    logger.info("Young -> 0")
    logger.info("Older -> 1")
    logger.info("Control -> 0")
    logger.info("Autism -> 1")

    return y


# ============================================================
# SHAP IMAGE
# ============================================================

def create_shap_bar_image(df_ranking, model_name, output_dir):
    """
    Create a horizontal bar image from global mean absolute SHAP ranking.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df_ranking.sort_values(
        by="Mean_ABS_SHAP",
        ascending=False
    ).head(50)

    # Reverse so the largest feature appears at the top
    df_plot = df_plot.iloc[::-1]

    plt.figure(figsize=(10, max(5, len(df_plot) * 0.35)))
    plt.barh(df_plot["Feature"], df_plot["Mean_ABS_SHAP"])
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("Feature")
    plt.title(f"XGBoost SHAP Summary - Model {model_name}")
    plt.tight_layout()

    safe_model_name = str(model_name).replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"SHAP_XGBoost_{safe_model_name}_summary.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved SHAP image: {output_path}")


# ============================================================
# XGBOOST + SHAP
# ============================================================

def train_xgboost_and_calculate_shap(
    X_train,
    y_train,
    X_explain,
    selected_indices,
    feature_names,
    model_params
):
    """
    Train one XGBoost model and calculate SHAP values.
    """

    max_index = len(feature_names) - 1
    invalid_indices = [idx for idx in selected_indices if idx < 0 or idx > max_index]

    if invalid_indices:
        raise ValueError(
            f"Invalid feature indices found: {invalid_indices}. "
            f"Valid range is 0 to {max_index}."
        )

    selected_feature_names = [feature_names[i] for i in selected_indices]

    X_train_selected = X_train[:, selected_indices]
    X_explain_selected = X_explain[:, selected_indices]

    model = XGBClassifier(
        n_estimators=int(model_params["n_estimators"]),
        max_depth=int(model_params["max_depth"]),
        learning_rate=float(model_params["learning_rate"]),
        subsample=float(model_params.get("subsample", 0.8)),
        colsample_bytree=float(model_params.get("colsample_bytree", 0.8)),
        eval_metric="logloss",
        random_state=int(model_params.get("random_state", 0))
    )

    model.fit(X_train_selected, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain_selected)

    shap_values = np.array(shap_values)

    # Binary classification usually gives: samples x features
    if shap_values.ndim == 2:
        shap_array = shap_values

    # Multiclass can give: samples x features x classes
    elif shap_values.ndim == 3:
        shap_array = np.mean(shap_values, axis=2)

    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

    return shap_array, selected_feature_names


def create_shap_values_dataframe(shap_values, feature_names, metadata_df):
    """
    Create sample-level SHAP values DataFrame.
    """

    df_shap = pd.DataFrame(shap_values, columns=feature_names)

    metadata_cols = [
        col for col in ["Participant", "Index", "Side", "Group"]
        if col in metadata_df.columns
    ]

    if metadata_cols:
        df_shap = pd.concat(
            [metadata_df[metadata_cols].reset_index(drop=True), df_shap],
            axis=1
        )

    df_shap["SUM"] = df_shap[feature_names].sum(axis=1)

    min_cols = df_shap[feature_names].idxmin(axis=1)
    max_cols = df_shap[feature_names].idxmax(axis=1)

    if "Group" in df_shap.columns:
        df_shap["Most Contributing Feat"] = np.where(
            df_shap["Group"].astype(str).isin(["0", "Control", "Young"]),
            min_cols,
            max_cols
        )
    else:
        df_shap["Most Contributing Feat"] = max_cols

    df_shap["Most Contributing Val"] = df_shap.apply(
        lambda row: row[row["Most Contributing Feat"]],
        axis=1
    )

    return df_shap


def create_global_ranking_dataframe(shap_values, feature_names):
    """
    Create global feature importance ranking using mean absolute SHAP.
    """

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    df_ranking = pd.DataFrame({
        "Feature": feature_names,
        "Mean_ABS_SHAP": mean_abs_shap
    })

    df_ranking = df_ranking.sort_values(
        by="Mean_ABS_SHAP",
        ascending=False
    ).reset_index(drop=True)

    df_ranking["SHAP_Rank"] = df_ranking.index + 1

    return df_ranking


# ============================================================
# MAIN FUNCTION CALLED BY PIPELINE
# ============================================================

def get_SHAP_data_XGBoost(
    scores_file,
    train_file,
    test_file,
    output_file,
    output_dir,
    labels_column="Group"
):
    """
    Main function to perform XGBoost SHAP analysis.
    """

    logger.info("Loading XGBoost scores file...")
    scores_excel = pd.ExcelFile(scores_file)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    shap_value_sheets = {}
    ranking_sheets = {}

    for sheet_name in scores_excel.sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")

        df_scores = pd.read_excel(scores_file, sheet_name=sheet_name)

        if df_scores.empty:
            logger.warning(f"No scores found for sheet: {sheet_name}")
            continue

        logger.info(f"Columns found in scores file: {df_scores.columns.tolist()}")

        feature_index_col = get_feature_index_column(df_scores)
        cv_col = get_cv_column(df_scores)

        logger.info(f"Using feature index column: {feature_index_col}")
        logger.info(f"Using CV column: {cv_col}")

        try:
            df_train = pd.read_excel(train_file, sheet_name=sheet_name)
            df_test = pd.read_excel(test_file, sheet_name=sheet_name)
        except ValueError:
            train_sheets = pd.ExcelFile(train_file).sheet_names
            test_sheets = pd.ExcelFile(test_file).sheet_names
            raise ValueError(
                f"Sheet '{sheet_name}' not found in train/test files.\n"
                f"Train sheets: {train_sheets}\n"
                f"Test sheets: {test_sheets}"
            )

        y_train = prepare_labels(df_train, labels_column)

        df_all = pd.concat([df_train, df_test], ignore_index=True)

        feature_names_all = get_feature_names(df_train)

        X_train_all = df_train.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"].values
        X_explain_all = df_all.loc[:, "Pelv_Angle_Y_MAX_SW":"Hip_Angle_Z_OHS"].values

        required_columns = [
            "Name_Model",
            "#Features",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "Test_Accuracy"
        ]

        missing_columns = [col for col in required_columns if col not in df_scores.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in scores file: {missing_columns}. "
                f"Available columns are: {df_scores.columns.tolist()}"
            )

        for _, row in df_scores.iterrows():

            model_name = row["Name_Model"]
            selected_indices = parse_selected_indices(row[feature_index_col])

            if len(selected_indices) == 0:
                logger.warning(
                    f"Skipping model {model_name} because selected indices are empty."
                )
                continue

            model_params = {
                "n_estimators": row["n_estimators"],
                "max_depth": row["max_depth"],
                "learning_rate": row["learning_rate"],
                "subsample": row["subsample"] if "subsample" in df_scores.columns else 0.8,
                "colsample_bytree": row["colsample_bytree"] if "colsample_bytree" in df_scores.columns else 0.8,
                "random_state": row["Random_State"] if "Random_State" in df_scores.columns else 0
            }

            shap_values, selected_feature_names = train_xgboost_and_calculate_shap(
                X_train=X_train_all,
                y_train=y_train,
                X_explain=X_explain_all,
                selected_indices=selected_indices,
                feature_names=feature_names_all,
                model_params=model_params
            )

            logger.info(
                f"SHAP completed for {sheet_name} | Model {model_name} | "
                f"Selected {len(selected_feature_names)} features"
            )

            df_shap_values = create_shap_values_dataframe(
                shap_values=shap_values,
                feature_names=selected_feature_names,
                metadata_df=df_all
            )

            df_ranking = create_global_ranking_dataframe(
                shap_values=shap_values,
                feature_names=selected_feature_names
            )

            df_ranking["Name_Model"] = model_name
            df_ranking["CV_Accuracy"] = row[cv_col]
            df_ranking["Test_Accuracy"] = row["Test_Accuracy"]
            df_ranking["#Features"] = row["#Features"]
            df_ranking["n_estimators"] = row["n_estimators"]
            df_ranking["max_depth"] = row["max_depth"]
            df_ranking["learning_rate"] = row["learning_rate"]

            if "Sensitivity" in df_scores.columns:
                df_ranking["Sensitivity"] = row["Sensitivity"]

            if "Specificity" in df_scores.columns:
                df_ranking["Specificity"] = row["Specificity"]

            if "F1" in df_scores.columns:
                df_ranking["F1"] = row["F1"]

            if "MCC" in df_scores.columns:
                df_ranking["MCC"] = row["MCC"]

            df_ranking["Selected_Indices"] = str(selected_indices)

            create_shap_bar_image(
                df_ranking=df_ranking,
                model_name=model_name,
                output_dir=output_dir
            )

            shap_sheet_name = f"SHAP_{sheet_name}_{model_name}"[:31]
            rank_sheet_name = f"Rank_{sheet_name}_{model_name}"[:31]

            shap_value_sheets[shap_sheet_name] = df_shap_values
            ranking_sheets[rank_sheet_name] = df_ranking

    if not shap_value_sheets and not ranking_sheets:
        raise ValueError("No SHAP results were generated. Check input files.")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        for sheet_name, df in shap_value_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        for sheet_name, df in ranking_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Saved XGBoost SHAP Excel results to: {output_file}")
    logger.info(f"Saved XGBoost SHAP images to: {output_dir}")