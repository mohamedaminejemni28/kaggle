import os
import sys
import json
import ast
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import torch
import torch.nn as nn


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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_list(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [value]

    return [value]


def load_checkpoint(path, device):
    """
    Load PyTorch checkpoint safely.
    """
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    return checkpoint


def encode_labels_from_checkpoint(y, label_classes):
    """
    Encode labels using the same class order saved in the checkpoint.
    """
    label_classes = [str(label) for label in label_classes]
    mapping = {label: index for index, label in enumerate(label_classes)}

    y_as_string = y.astype(str)
    y_encoded = y_as_string.map(mapping)

    if y_encoded.isna().any():
        missing = sorted(y_as_string[y_encoded.isna()].unique())
        raise ValueError(f"Unknown labels found: {missing}")

    return y_encoded.to_numpy(dtype=int)


def transform_with_checkpoint(X_raw, imputer_statistics, scaler_mean, scaler_scale):
    """
    Apply the saved median imputer and StandardScaler from the checkpoint.
    """
    X = np.asarray(X_raw, dtype=np.float32).copy()

    imputer_statistics = np.asarray(imputer_statistics, dtype=np.float32)
    scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float32)

    # Replace NaN values with saved median values
    nan_rows, nan_cols = np.where(np.isnan(X))

    if len(nan_rows) > 0:
        X[nan_rows, nan_cols] = imputer_statistics[nan_cols]

    # Avoid division by zero
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

    X_scaled = (X - scaler_mean) / scaler_scale

    return X_scaled.astype(np.float32)


def classification_metrics(y_true, probabilities):
    predictions = probabilities.argmax(axis=1)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
    else:
        specificity = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Sensitivity": recall_score(y_true, predictions, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, predictions, zero_division=0),
    }

    try:
        metrics["ROC_AUC"] = roc_auc_score(y_true, probabilities[:, 1])
    except ValueError:
        metrics["ROC_AUC"] = np.nan

    return {key: float(value) for key, value in metrics.items()}, cm


def score_model(y_true, probabilities):
    """
    Use balanced accuracy when both classes exist.
    Otherwise use accuracy.
    """
    predictions = probabilities.argmax(axis=1)

    if len(np.unique(y_true)) >= 2:
        return balanced_accuracy_score(y_true, predictions)

    return accuracy_score(y_true, predictions)


# ============================================================
# AutoencoderClassifier Model
# Same architecture as Phase 4.2
# ============================================================

class GaitAutoencoderClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes=2,
        encoder_layers=None,
        latent_dim=8,
        classifier_layers=None,
        dropout=0.2,
    ):
        super().__init__()

        if encoder_layers is None:
            encoder_layers = [64, 32]
        if classifier_layers is None:
            classifier_layers = [32]

        encoder = []
        input_dim = num_features
        for hidden_dim in encoder_layers:
            encoder.append(nn.Linear(input_dim, hidden_dim))
            encoder.append(nn.ReLU())
            encoder.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        encoder.append(nn.Linear(input_dim, latent_dim))
        encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        input_dim = latent_dim
        for hidden_dim in reversed(encoder_layers):
            decoder.append(nn.Linear(input_dim, hidden_dim))
            decoder.append(nn.ReLU())
            decoder.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        decoder.append(nn.Linear(input_dim, num_features))
        self.decoder = nn.Sequential(*decoder)

        classifier = []
        input_dim = latent_dim
        for hidden_dim in classifier_layers:
            classifier.append(nn.Linear(input_dim, hidden_dim))
            classifier.append(nn.ReLU())
            classifier.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        classifier.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def encode(self, x):
        return self.encoder(x)

    def reconstruct(self, x):
        return self.decoder(self.encode(x))

    def forward(self, x):
        return self.classifier(self.encode(x))


def build_model(model_config, num_features, num_classes=2):
    return GaitAutoencoderClassifier(
        num_features=num_features,
        num_classes=num_classes,
        encoder_layers=model_config["encoder_layers"],
        latent_dim=model_config["latent_dim"],
        classifier_layers=model_config["classifier_layers"],
        dropout=model_config["dropout"],
    )


@torch.no_grad()
def predict_probabilities(model, X, device):
    model.eval()

    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    logits = model(tensor)

    return torch.softmax(logits, dim=1).cpu().numpy()


# ============================================================
# Permutation Importance
# ============================================================

def permutation_importance(
    model,
    X_raw,
    y_true,
    feature_names,
    imputer_statistics,
    scaler_mean,
    scaler_scale,
    device,
    n_repeats=30,
    seed=42,
):
    """
    Compute permutation importance.

    Importance = base score - score after shuffling one feature.
    """
    rng = np.random.default_rng(seed)

    X_scaled = transform_with_checkpoint(
        X_raw,
        imputer_statistics,
        scaler_mean,
        scaler_scale,
    )

    base_probabilities = predict_probabilities(model, X_scaled, device)
    base_score = score_model(y_true, base_probabilities)
    base_metrics, base_cm = classification_metrics(y_true, base_probabilities)

    logger.info(f"Base balanced/accuracy score: {base_score:.4f}")

    detailed_rows = []
    summary_rows = []

    for feature_index, feature_name in enumerate(feature_names):
        repeat_importances = []

        for repeat in range(1, n_repeats + 1):
            X_permuted = np.asarray(X_raw, dtype=np.float32).copy()

            shuffled_values = X_permuted[:, feature_index].copy()
            rng.shuffle(shuffled_values)

            X_permuted[:, feature_index] = shuffled_values

            X_permuted_scaled = transform_with_checkpoint(
                X_permuted,
                imputer_statistics,
                scaler_mean,
                scaler_scale,
            )

            permuted_probabilities = predict_probabilities(
                model,
                X_permuted_scaled,
                device,
            )

            permuted_score = score_model(y_true, permuted_probabilities)
            importance = base_score - permuted_score

            repeat_importances.append(importance)

            detailed_rows.append({
                "Feature": feature_name,
                "Feature_Index": feature_index,
                "Repeat": repeat,
                "Base_Score": base_score,
                "Permuted_Score": permuted_score,
                "Importance": importance,
            })

        summary_rows.append({
            "Feature": feature_name,
            "Feature_Index": feature_index,
            "Mean_Importance": float(np.mean(repeat_importances)),
            "Std_Importance": float(np.std(repeat_importances)),
            "Min_Importance": float(np.min(repeat_importances)),
            "Max_Importance": float(np.max(repeat_importances)),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="Mean_Importance",
        ascending=False,
    ).reset_index(drop=True)

    summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

    detailed_df = pd.DataFrame(detailed_rows)

    return summary_df, detailed_df, base_metrics, base_cm


def save_importance_plot(summary_df, output_dir, top_n=15):
    """
    Save top feature importance plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_df = summary_df.head(top_n).copy()
    plot_df = plot_df.sort_values("Mean_Importance", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["Feature"], plot_df["Mean_Importance"])
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.title("Top AutoencoderClassifier Feature Importance")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "AutoencoderClassifier_permutation_importance_top_features.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

    logger.info(f"Saved importance plot to: {plot_file}")

    return plot_file


def save_confusion_matrix_plot(cm, label_classes, output_dir):
    """
    Save confusion matrix plot if available.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))
    image = plt.imshow(cm, cmap="Blues")

    plt.title("AutoencoderClassifier Evaluation Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    plt.xticks([0, 1], label_classes)
    plt.yticks([0, 1], label_classes)

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                int(cm[row, column]),
                ha="center",
                va="center",
                color="black",
                fontsize=13,
            )

    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "AutoencoderClassifier_evaluation_confusion_matrix.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()

    logger.info(f"Saved confusion matrix plot to: {plot_file}")

    return plot_file


# ============================================================
# Main
# ============================================================

def main():
    logger.info(">>>>>>>>>> Starting AutoencoderClassifier Phase 4.4: AutoencoderClassifier Interpretation <<<<<<<<<<")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    AutoencoderClassifier_config = config.get("AutoencoderClassifier", {})

    seed = int(AutoencoderClassifier_config.get("random_state", 42))
    set_seed(seed)

    group_column = config.get("SFS", {}).get("labels_column", "Group")
    participant_column = AutoencoderClassifier_config.get("participant_column", "Participant")
    strict_subject_split = bool(AutoencoderClassifier_config.get("strict_subject_split", True))

    n_repeats = int(AutoencoderClassifier_config.get("permutation_repeats", 30))

    train_file = os.path.join(
        PROJECT_ROOT,
        config["split_datasets"]["output_train_file"],
    )

    test_file = os.path.join(
        PROJECT_ROOT,
        config["split_datasets"]["output_test_file"],
    )

    top3_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_top3_AutoencoderClassifier"],
    )

    checkpoint_file = os.path.join(
        PROJECT_ROOT,
        "results",
        "AutoencoderClassifier_checkpoints",
        "best_AutoencoderClassifier_model.pt",
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["output_file_AutoencoderClassifier"],
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        config["data"]["output_dir_AutoencoderClassifier"],
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for label, path in {
        "Train": train_file,
        "Test": test_file,
        "Top3": top3_file,
        "Checkpoint": checkpoint_file,
    }.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

        logger.info(f"{label}: {path}")

    # ========================================================
    # Load top 3 file to get dataset sheet
    # ========================================================

    excel = pd.ExcelFile(top3_file)

    if "Top3_Overall" in excel.sheet_names:
        top3_df = pd.read_excel(top3_file, sheet_name="Top3_Overall")
    else:
        top3_df = pd.read_excel(top3_file, sheet_name=excel.sheet_names[0])

    if top3_df.empty:
        raise ValueError("Top3 AutoencoderClassifier file is empty.")

    sheet_name = str(top3_df.iloc[0]["Dataset_Sheet"])

    logger.info(f"Using dataset sheet: {sheet_name}")

    # ========================================================
    # Load checkpoint
    # ========================================================

    checkpoint = load_checkpoint(checkpoint_file, device)

    model_config = checkpoint["model_config"]
    feature_names = [str(name) for name in checkpoint["feature_names"]]
    label_classes = [str(label) for label in checkpoint["label_classes"]]

    imputer_statistics = checkpoint["imputer_statistics"]
    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]

    logger.info(f"Loaded best AutoencoderClassifier config: {model_config}")
    logger.info(f"Number of selected features: {len(feature_names)}")
    logger.info(f"Selected features: {feature_names}")

    # ========================================================
    # Load train/test data
    # ========================================================

    train_df = pd.read_excel(train_file, sheet_name=sheet_name)
    test_df = pd.read_excel(test_file, sheet_name=sheet_name)

    for name, frame in {"train": train_df, "test": test_df}.items():
        if group_column not in frame.columns:
            raise ValueError(f"{group_column!r} is missing from {name} data.")

        if participant_column not in frame.columns:
            raise ValueError(
                f"{participant_column!r} is required for participant-aware validation."
            )

    # ========================================================
    # Clean participant IDs - IMPORTANT FIX
    # ========================================================

    train_df[participant_column] = (
        train_df[participant_column]
        .fillna("UNKNOWN_PARTICIPANT")
        .astype(str)
        .str.strip()
    )

    test_df[participant_column] = (
        test_df[participant_column]
        .fillna("UNKNOWN_PARTICIPANT")
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # Participant overlap check - FIXED
    # ========================================================

    train_participants = set(train_df[participant_column])
    test_participants = set(test_df[participant_column])

    overlap = sorted(train_participants & test_participants)

    logger.info(
        f"Train rows / participants: {len(train_df)} / "
        f"{train_df[participant_column].nunique()}"
    )

    logger.info(
        f"Test rows / participants: {len(test_df)} / "
        f"{test_df[participant_column].nunique()}"
    )

    logger.info(f"Overlapping participants: {overlap}")

    if overlap and strict_subject_split:
        before = len(test_df)

        test_df = test_df.loc[
            ~test_df[participant_column].isin(overlap)
        ].reset_index(drop=True)

        logger.info(f"Removed {before - len(test_df)} overlapping test rows.")
        logger.info(f"Test rows after overlap removal: {len(test_df)}")

    elif overlap:
        logger.warning(
            "Train/test participant overlap remains. Test metrics may be optimistic."
        )

    if len(test_df) == 0:
        logger.warning(
            "Independent test set is empty after removing overlapping participants. "
            "Permutation importance will use training data."
        )

    # ========================================================
    # Choose interpretation dataset
    # ========================================================

    if len(test_df) > 0:
        interpretation_df = test_df.copy()
        interpretation_source = "independent_test"
    else:
        interpretation_df = train_df.copy()
        interpretation_source = "train_data"

        logger.warning(
            "Independent test set is empty. "
            "Permutation importance will be computed on training data."
        )

    missing_features = [
        feature for feature in feature_names
        if feature not in interpretation_df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing selected features in data: {missing_features}")

    X_raw = interpretation_df[feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    y_true = encode_labels_from_checkpoint(
        interpretation_df[group_column],
        label_classes,
    )

    logger.info(f"Interpretation source: {interpretation_source}")
    logger.info(f"Interpretation rows: {len(interpretation_df)}")
    logger.info(f"Class counts: {np.bincount(y_true)}")

    # ========================================================
    # Build model and load weights
    # ========================================================

    model = build_model(
        model_config=model_config,
        num_features=len(feature_names),
        num_classes=len(label_classes),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ========================================================
    # Compute permutation importance
    # ========================================================

    summary_df, detailed_df, base_metrics, base_cm = permutation_importance(
        model=model,
        X_raw=X_raw,
        y_true=y_true,
        feature_names=feature_names,
        imputer_statistics=imputer_statistics,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        device=device,
        n_repeats=n_repeats,
        seed=seed,
    )

    base_metrics_df = pd.DataFrame([{
        "Model": "AutoencoderClassifier",
        "Interpretation_Source": interpretation_source,
        "Rows": len(interpretation_df),
        "N_Repeats": n_repeats,
        "Model_Config": json.dumps(model_config, sort_keys=True),
        "Feature_Names": json.dumps(feature_names),
        **base_metrics,
    }])

    prediction_probabilities = predict_probabilities(
        model,
        transform_with_checkpoint(
            X_raw,
            imputer_statistics,
            scaler_mean,
            scaler_scale,
        ),
        device,
    )

    predictions_df = interpretation_df[[participant_column, group_column]].copy()
    predictions_df["True_Label"] = interpretation_df[group_column].astype(str).to_numpy()
    predictions_df["Predicted_Class_Index"] = prediction_probabilities.argmax(axis=1)
    predictions_df["Predicted_Label"] = [
        label_classes[index] for index in prediction_probabilities.argmax(axis=1)
    ]
    predictions_df["Probability_Class_0"] = prediction_probabilities[:, 0]
    predictions_df["Probability_Class_1"] = prediction_probabilities[:, 1]

    # ========================================================
    # Save Excel outputs
    # ========================================================

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        base_metrics_df.to_excel(writer, sheet_name="Base_Metrics", index=False)
        summary_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
        detailed_df.to_excel(writer, sheet_name="Permutation_Details", index=False)
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

    logger.info(f"Saved AutoencoderClassifier interpretation Excel to: {output_file}")

    # ========================================================
    # Save plots
    # ========================================================

    save_importance_plot(
        summary_df=summary_df,
        output_dir=output_dir,
        top_n=min(15, len(summary_df)),
    )

    save_confusion_matrix_plot(
        cm=base_cm,
        label_classes=label_classes,
        output_dir=output_dir,
    )

    logger.info("Top AutoencoderClassifier important features:")

    for _, row in summary_df.head(10).iterrows():
        logger.info(
            f"Rank {int(row['Rank'])} | "
            f"{row['Feature']} | "
            f"Importance = {row['Mean_Importance']:.4f}"
        )

    logger.info(">>>>>>>>>> Completed AutoencoderClassifier Phase 4.4: AutoencoderClassifier Interpretation <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in AutoencoderClassifier Phase 4.4: {e}")
        raise e
