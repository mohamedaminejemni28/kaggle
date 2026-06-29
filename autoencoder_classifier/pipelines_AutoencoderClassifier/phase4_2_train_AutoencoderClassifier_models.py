import os
import sys
import ast
import json
import copy
import math
import random
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
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
from torch.utils.data import DataLoader, TensorDataset


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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def parse_list(value):
    """
    Parse values like:
    "['feature1', 'feature2']"
    "[1, 2, 3]"
    """
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else [parsed]

    return [value]


def fit_preprocessor(X):
    """
    Fit imputer and scaler only on training fold.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled.astype(np.float32), imputer, scaler


def transform_preprocessor(X, imputer, scaler):
    """
    Apply fitted imputer and scaler.
    """
    return scaler.transform(imputer.transform(X)).astype(np.float32)


def class_weight_tensor(y, device):
    """
    Balanced class weights for small datasets.
    """
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    full_weights = np.ones(2, dtype=np.float32)
    full_weights[classes] = weights.astype(np.float32)

    return torch.tensor(full_weights, dtype=torch.float32, device=device)


def classification_metrics(y_true, probabilities):
    """
    Compute binary classification metrics.
    """
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


# ============================================================
# AutoencoderClassifier Model
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


# ============================================================
# Training Utilities
# ============================================================

def make_loader(X, y, batch_size, shuffle=True, seed=42):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def predict_probabilities(model, X, device):
    model.eval()

    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    logits = model(tensor)

    return torch.softmax(logits, dim=1).cpu().numpy()


def train_with_early_stopping(
    X_train,
    y_train,
    X_valid,
    y_valid,
    model_config,
    max_epochs,
    patience,
    seed,
    device,
):
    set_seed(seed)

    model = build_model(
        model_config=model_config,
        num_features=X_train.shape[1],
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor(y_train, device),
    )
    reconstruction_criterion = nn.MSELoss()
    reconstruction_weight = float(model_config.get("reconstruction_weight", 0.2))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )

    loader = make_loader(
        X_train,
        y_train,
        batch_size=model_config["batch_size"],
        shuffle=True,
        seed=seed,
    )

    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.long).to(device)

    best_state = None
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "valid_loss": [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            reconstruction = model.reconstruct(xb)
            classification_loss = criterion(logits, yb)
            reconstruction_loss = reconstruction_criterion(reconstruction, xb)
            loss = classification_loss + reconstruction_weight * reconstruction_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(xb)

        train_loss = running_loss / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            valid_logits = model(X_valid_tensor)
            valid_reconstruction = model.reconstruct(X_valid_tensor)
            valid_classification_loss = criterion(valid_logits, y_valid_tensor)
            valid_reconstruction_loss = reconstruction_criterion(valid_reconstruction, X_valid_tensor)
            valid_loss = (valid_classification_loss + reconstruction_weight * valid_reconstruction_loss).item()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, history


def train_full_data(
    X,
    y,
    model_config,
    epochs,
    seed,
    device,
):
    set_seed(seed)

    model = build_model(
        model_config=model_config,
        num_features=X.shape[1],
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_tensor(y, device),
    )
    reconstruction_criterion = nn.MSELoss()
    reconstruction_weight = float(model_config.get("reconstruction_weight", 0.2))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )

    loader = make_loader(
        X,
        y,
        batch_size=model_config["batch_size"],
        shuffle=True,
        seed=seed,
    )

    losses = []

    for _ in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            reconstruction = model.reconstruct(xb)
            classification_loss = criterion(logits, yb)
            reconstruction_loss = reconstruction_criterion(reconstruction, xb)
            loss = classification_loss + reconstruction_weight * reconstruction_loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(xb)

        losses.append(running_loss / len(loader.dataset))

    return model, losses


# ============================================================
# Candidate Generation
# ============================================================

def generate_AutoencoderClassifier_candidates(feature_sets_df, AutoencoderClassifier_config, seed=42):
    """
    Generate AutoencoderClassifier candidate models.
    Each row = one AutoencoderClassifier experiment.
    """
    random.seed(seed)
    np.random.seed(seed)

    feature_counts = AutoencoderClassifier_config.get("feature_counts", None)

    if feature_counts is not None:
        feature_sets_df = feature_sets_df[
            feature_sets_df["#Features"].astype(int).isin(feature_counts)
        ].reset_index(drop=True)

    if feature_sets_df.empty:
        raise ValueError("No feature sets available for AutoencoderClassifier training.")

    encoder_layers_list = AutoencoderClassifier_config.get(
        "encoder_layers",
        [[32], [64, 32], [128, 64]],
    )
    latent_dim_list = AutoencoderClassifier_config.get("latent_dim", [4, 8, 16])
    classifier_layers_list = AutoencoderClassifier_config.get("classifier_layers", [[16], [32], [32, 16]])
    dropout_list = AutoencoderClassifier_config.get("dropout", [0.1, 0.2, 0.3])
    reconstruction_weight_list = AutoencoderClassifier_config.get("reconstruction_weight", [0.1, 0.2, 0.5])
    learning_rate_list = AutoencoderClassifier_config.get("learning_rate", [0.001, 0.0005, 0.0001])
    batch_size_list = AutoencoderClassifier_config.get("batch_size", [4, 8, 16])
    weight_decay_list = AutoencoderClassifier_config.get("weight_decay", [0.0, 0.0001, 0.001])

    max_candidates = AutoencoderClassifier_config.get("max_candidates", 1000)

    candidate_rows = []
    candidate_id = 1

    for _, feature_row in feature_sets_df.iterrows():
        feature_names = [str(name) for name in parse_list(feature_row["Feature_Names"])]

        for encoder_layers, latent_dim, classifier_layers, dropout, reconstruction_weight, lr, batch_size, weight_decay in itertools.product(
            encoder_layers_list,
            latent_dim_list,
            classifier_layers_list,
            dropout_list,
            reconstruction_weight_list,
            learning_rate_list,
            batch_size_list,
            weight_decay_list,
        ):
            model_config = {
                "architecture": "AutoencoderClassifier",
                "encoder_layers": encoder_layers,
                "latent_dim": int(latent_dim),
                "classifier_layers": classifier_layers,
                "dropout": float(dropout),
                "reconstruction_weight": float(reconstruction_weight),
                "learning_rate": float(lr),
                "batch_size": int(batch_size),
                "weight_decay": float(weight_decay),
            }

            candidate_rows.append({
                "Candidate_ID": candidate_id,
                "Dataset_Sheet": feature_row["Dataset_Sheet"],
                "Model": "AutoencoderClassifier",
                "#Features": int(feature_row["#Features"]),
                "Feature_Names": json.dumps(feature_names),
                "Model_Config": json.dumps(model_config, sort_keys=True),
            })

            candidate_id += 1

    candidates_df = pd.DataFrame(candidate_rows)

    if max_candidates is not None and int(max_candidates) > 0:
        if len(candidates_df) > int(max_candidates):
            candidates_df = candidates_df.sample(
                n=int(max_candidates),
                random_state=seed,
            ).sort_values("Candidate_ID").reset_index(drop=True)

            candidates_df["Candidate_ID"] = range(1, len(candidates_df) + 1)

    return candidates_df


# ============================================================
# Main
# ============================================================

def main():
    logger.info(">>>>>>>>>> Starting AutoencoderClassifier Phase 4.2: Training AutoencoderClassifier Models <<<<<<<<<<")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    config = read_yaml(config_path)

    AutoencoderClassifier_config = config.get("AutoencoderClassifier", {})

    seed = int(AutoencoderClassifier_config.get("random_state", 42))
    set_seed(seed)

    group_column = config.get("SFS", {}).get("labels_column", "Group")
    participant_column = AutoencoderClassifier_config.get("participant_column", "Participant")

    strict_subject_split = bool(AutoencoderClassifier_config.get("strict_subject_split", True))
    n_splits = int(AutoencoderClassifier_config.get("n_splits", 5))
    max_epochs = int(AutoencoderClassifier_config.get("max_epochs", 150))
    patience = int(AutoencoderClassifier_config.get("patience", 20))

    train_file = os.path.join(
        PROJECT_ROOT,
        config["split_datasets"]["output_train_file"],
    )

    test_file = os.path.join(
        PROJECT_ROOT,
        config["split_datasets"]["output_test_file"],
    )

    feature_sets_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["sfs_results_excel_file_AutoencoderClassifier"],
    )

    output_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["combination_results_file_AutoencoderClassifier"],
    )

    top3_output_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_top3_AutoencoderClassifier"],
    )

    final_scores_file = os.path.join(
        PROJECT_ROOT,
        config["data"]["scores_file_AutoencoderClassifier"],
    )

    checkpoint_dir = os.path.join(PROJECT_ROOT, "results", "AutoencoderClassifier_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(checkpoint_dir, "best_AutoencoderClassifier_model.pt")

    for label, path in {
        "Train": train_file,
        "Test": test_file,
        "Feature sets": feature_sets_file,
    }.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

        logger.info(f"{label}: {path}")

    # ========================================================
    # Load feature sets
    # ========================================================

    feature_sets_df = pd.read_excel(feature_sets_file)

    required_columns = {"Dataset_Sheet", "#Features", "Feature_Names"}
    missing_columns = required_columns - set(feature_sets_df.columns)

    if missing_columns:
        raise ValueError(f"Missing feature-set columns: {sorted(missing_columns)}")

    sheet_name = str(feature_sets_df.iloc[0]["Dataset_Sheet"])

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
    # Participant overlap check
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
            "The code will continue and final test metrics will be NaN."
        )

    # ========================================================
    # Encode labels
    # ========================================================

    label_encoder = LabelEncoder()
    y_train_all = label_encoder.fit_transform(train_df[group_column]).astype(int)

    if len(test_df) == 0:
        y_test = np.array([], dtype=int)
    else:
        y_test = label_encoder.transform(test_df[group_column]).astype(int)

    groups_all = (
        train_df[participant_column]
        .fillna("UNKNOWN_PARTICIPANT")
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    logger.info(f"Classes: {dict(enumerate(label_encoder.classes_))}")
    logger.info(f"Train class counts: {np.bincount(y_train_all)}")

    if len(y_test) == 0:
        logger.warning(
            "Independent test set is empty after removing overlapping participants."
        )
    else:
        logger.info(f"Independent test class counts: {np.bincount(y_test)}")

    # ========================================================
    # Generate candidates
    # ========================================================

    candidates_df = generate_AutoencoderClassifier_candidates(
        feature_sets_df=feature_sets_df,
        AutoencoderClassifier_config=AutoencoderClassifier_config,
        seed=seed,
    )

    logger.info(f"Number of AutoencoderClassifier candidates: {len(candidates_df)}")

    # ========================================================
    # Participant-aware CV
    # ========================================================

    unique_groups = pd.Series(groups_all).astype(str).unique()
    min_class_count = np.min(np.bincount(y_train_all))
    local_n_splits = min(n_splits, len(unique_groups), int(min_class_count))

    if local_n_splits < 2:
        raise ValueError(
            f"Not enough groups/classes for StratifiedGroupKFold. "
            f"local_n_splits={local_n_splits}"
        )

    logger.info(f"Using StratifiedGroupKFold with n_splits={local_n_splits}")

    cv = StratifiedGroupKFold(
        n_splits=local_n_splits,
        shuffle=True,
        random_state=seed,
    )

    all_results = []

    total_candidates = len(candidates_df)

    for _, candidate in candidates_df.iterrows():
        candidate_id = int(candidate["Candidate_ID"])

        model_config = json.loads(candidate["Model_Config"])
        feature_names = json.loads(candidate["Feature_Names"])

        missing = [name for name in feature_names if name not in train_df.columns]

        if missing:
            raise ValueError(f"Features missing from training data: {missing}")

        X_raw = train_df[feature_names].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float32)

        fold_metrics = []
        best_epochs = []

        for fold, (train_index, valid_index) in enumerate(
            cv.split(X_raw, y_train_all, groups_all),
            start=1,
        ):
            X_fold_train, imputer, scaler = fit_preprocessor(X_raw[train_index])

            X_fold_valid = transform_preprocessor(
                X_raw[valid_index],
                imputer,
                scaler,
            )

            model, best_epoch, _ = train_with_early_stopping(
                X_train=X_fold_train,
                y_train=y_train_all[train_index],
                X_valid=X_fold_valid,
                y_valid=y_train_all[valid_index],
                model_config=model_config,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed + fold + candidate_id,
                device=device,
            )

            probabilities = predict_probabilities(
                model=model,
                X=X_fold_valid,
                device=device,
            )

            metrics, _ = classification_metrics(
                y_train_all[valid_index],
                probabilities,
            )

            fold_metrics.append(metrics)
            best_epochs.append(best_epoch)

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

        result_row = {
            "Candidate_ID": candidate_id,
            "Dataset_Sheet": candidate["Dataset_Sheet"],
            "Model": "AutoencoderClassifier",
            "Architecture": "AutoencoderClassifier",
            "#Features": len(feature_names),
            "Feature_Names": json.dumps(feature_names),
            "Model_Config": json.dumps(model_config, sort_keys=True),
            "Best_Epoch_Median": int(np.median(best_epochs)),
            "Best_Epochs": json.dumps(best_epochs),
        }

        for metric_name in fold_metrics[0]:
            values = [metrics[metric_name] for metrics in fold_metrics]

            result_row[f"CV_{metric_name}_Mean"] = float(np.nanmean(values))
            result_row[f"CV_{metric_name}_Std"] = float(np.nanstd(values))

        all_results.append(result_row)

        logger.info(
            f"[{candidate_id:04d}/{total_candidates}] "
            f"{len(feature_names):2d} features | "
            f"encoder={model_config['encoder_layers']} | "
            f"latent={model_config['latent_dim']} | "
            f"dropout={model_config['dropout']} | recon={model_config['reconstruction_weight']} | "
            f"balanced accuracy={result_row['CV_Balanced_Accuracy_Mean']:.3f}"
        )

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by=[
            "CV_Balanced_Accuracy_Mean",
            "CV_F1_Mean",
            "CV_ROC_AUC_Mean",
            "CV_Balanced_Accuracy_Std",
            "#Features",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_excel(output_file, index=False)

    top3_df = results_df.head(3).copy()
    top3_df.to_excel(top3_output_file, index=False)

    logger.info(f"Saved all AutoencoderClassifier results to: {output_file}")
    logger.info(f"Saved Top 3 AutoencoderClassifier results to: {top3_output_file}")

    # ========================================================
    # Final training using best model
    # ========================================================

    best = results_df.iloc[0]
    best_config = json.loads(best["Model_Config"])
    best_features = json.loads(best["Feature_Names"])
    final_epochs = max(5, int(best["Best_Epoch_Median"]))

    logger.info(f"Best AutoencoderClassifier config: {best_config}")
    logger.info(f"Best feature count: {len(best_features)}")
    logger.info(f"Final epochs: {final_epochs}")

    X_train_raw = train_df[best_features].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float32)

    X_train_final, final_imputer, final_scaler = fit_preprocessor(X_train_raw)

    final_model, final_losses = train_full_data(
        X=X_train_final,
        y=y_train_all,
        model_config=best_config,
        epochs=final_epochs,
        seed=seed,
        device=device,
    )

    final_result = {
        "Model": "AutoencoderClassifier",
        "Architecture": "AutoencoderClassifier",
        "#Features": len(best_features),
        "Feature_Names": json.dumps(best_features),
        "Model_Config": json.dumps(best_config, sort_keys=True),
        "Final_Epochs": final_epochs,
        "Independent_Test_Rows": len(test_df),
        "Excluded_Overlap_Participants": json.dumps(
            overlap if strict_subject_split else []
        ),
    }

    test_cm = None

    if len(test_df) == 0:
        logger.warning(
            "Final test metrics were not computed because the independent test set is empty."
        )

        final_result.update({
            "Test_Accuracy": np.nan,
            "Test_Balanced_Accuracy": np.nan,
            "Test_Precision": np.nan,
            "Test_Sensitivity": np.nan,
            "Test_Specificity": np.nan,
            "Test_F1": np.nan,
            "Test_ROC_AUC": np.nan,
        })

    else:
        X_test_raw = test_df[best_features].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=np.float32)

        X_test_final = transform_preprocessor(
            X_test_raw,
            final_imputer,
            final_scaler,
        )

        test_probabilities = predict_probabilities(
            model=final_model,
            X=X_test_final,
            device=device,
        )

        test_metrics, test_cm = classification_metrics(
            y_test,
            test_probabilities,
        )

        final_result.update({
            f"Test_{key}": value for key, value in test_metrics.items()
        })

    final_df = pd.DataFrame([final_result])
    final_df.to_excel(final_scores_file, index=False)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "model_config": best_config,
            "num_features": len(best_features),
            "feature_names": best_features,
            "label_classes": label_encoder.classes_.tolist(),
            "imputer_statistics": final_imputer.statistics_,
            "scaler_mean": final_scaler.mean_,
            "scaler_scale": final_scaler.scale_,
            "final_epochs": final_epochs,
            "seed": seed,
            "strict_subject_split": strict_subject_split,
        },
        checkpoint_file,
    )

    logger.info(f"Saved final AutoencoderClassifier scores to: {final_scores_file}")
    logger.info(f"Saved AutoencoderClassifier checkpoint to: {checkpoint_file}")

    # ========================================================
    # Final plot
    # ========================================================

    if test_cm is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(final_losses)
        axes[0].set_title("AutoencoderClassifier Final Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)

        image = axes[1].imshow(test_cm, cmap="Blues")
        axes[1].set_title("Independent Test Confusion Matrix")
        axes[1].set_xlabel("Predicted class")
        axes[1].set_ylabel("True class")
        axes[1].set_xticks([0, 1], label_encoder.classes_)
        axes[1].set_yticks([0, 1], label_encoder.classes_)

        for row in range(2):
            for column in range(2):
                axes[1].text(
                    column,
                    row,
                    int(test_cm[row, column]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="black",
                )

        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    else:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(final_losses)
        ax.set_title("AutoencoderClassifier Final Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)

    plt.tight_layout()

    plot_file = os.path.join(PROJECT_ROOT, "results", "AutoencoderClassifier_final_training_plot.png")
    plt.savefig(plot_file)
    plt.close()

    logger.info(f"Saved AutoencoderClassifier final plot to: {plot_file}")

    logger.info(">>>>>>>>>> Completed AutoencoderClassifier Phase 4.2: Training AutoencoderClassifier Models <<<<<<<<<<")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred in AutoencoderClassifier Phase 4.2: {e}")
        raise e
