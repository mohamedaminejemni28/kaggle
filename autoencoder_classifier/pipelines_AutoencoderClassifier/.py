from pathlib import Path

import mlflow
import pandas as pd


# ============================================================
# 1) MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Young_and_Old")


# ============================================================
# 2) FILES
# ============================================================

files = {
    # Config
    "config_young_old": "configs/config_young_old_2024.yaml",
    "config_current": "configs/config.yaml",

    # SVM / RBF files
    "svm_phase1_JA_TS": "data/processed/young_old_2024_JA_TS_features.xlsx",
    "svm_phase2_NH_correlated": "data/processed/young_old_2024_NH_correlated_features.xlsx",
    "svm_train_file": "data/processed/young_old_2024_train.xlsx",
    "svm_test_file": "data/processed/young_old_2024_test.xlsx",
    "svm_sfs_results": "src/gaitml/features/young_old_2024_phase4_1_SFS_results.xlsx",
    "svm_step2_filtered": "young_old_2024_Step_2_Results_Filtered.xlsx",
    "svm_scores_rbf": "young_old_2024_Scores_RBF.xlsx",
    "svm_scores_rbf_top3": "young_old_2024_Scores_RBF_Top3.xlsx",
    "svm_shap_excel": "output2/young_old_2024_SHAP/young_old_2024_SHAP_data.xlsx",

    # XGBoost files
    "xgb_phase1_JA_TS": "results/young_old_2024_JA_TS_features.xlsx",
    "xgb_phase2_NH_correlated": "results/young_old_2024_NH_correlated_features.xlsx",
    "xgb_train_file": "results/young_old_2024_train.xlsx",
    "xgb_test_file": "results/young_old_2024_test.xlsx",
    "xgb_sfs_results": "results/young_old_2024_phase4_1_SFS_results_XGBoost.xlsx",
    "xgb_step2_results": "results/young_old_2024_Step_2_Results_XGBoost.xlsx",
    "xgb_scores": "results/young_old_2024_Scores_XGBoost.xlsx",
    "xgb_scores_top3": "results/young_old_2024_Scores_XGBoost_Top3.xlsx",
    "xgb_shap_excel": "output-young_old/young_old_2024_SHAP_data_XGBoost.xlsx",
}

svm_shap_dir = Path("output2/young_old_2024_SHAP")
xgb_shap_dir = Path("output-young_old/young_old_2024_SHAP")


# ============================================================
# 3) HELPER FUNCTIONS
# ============================================================

def read_top3_file(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Top3 file not found: {file_path}")
        return None

    excel = pd.ExcelFile(path)
    sheet_name = excel.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet_name)

    print(f"Loaded Top3 file: {file_path}")
    print(f"Sheet used: {sheet_name}")
    print(df.head())

    return df


def log_top3_metrics(df_top3, model_prefix):
    if df_top3 is None or df_top3.empty:
        print(f"No Top3 data available for {model_prefix}")
        return

    best_row = df_top3.iloc[0]

    # Best model params
    mlflow.log_param(f"{model_prefix}_best_model_name", str(best_row.get("Name_Model", "")))
    mlflow.log_param(f"{model_prefix}_best_model_num_features", int(best_row.get("#Features", 0)))

    if "C_Value" in df_top3.columns:
        mlflow.log_param(f"{model_prefix}_best_C_value", float(best_row["C_Value"]))

    if "Gamma_Value" in df_top3.columns:
        mlflow.log_param(f"{model_prefix}_best_gamma_value", float(best_row["Gamma_Value"]))

    if "n_estimators" in df_top3.columns:
        mlflow.log_param(f"{model_prefix}_best_n_estimators", int(best_row["n_estimators"]))

    if "max_depth" in df_top3.columns:
        mlflow.log_param(f"{model_prefix}_best_max_depth", int(best_row["max_depth"]))

    if "learning_rate" in df_top3.columns:
        mlflow.log_param(f"{model_prefix}_best_learning_rate", float(best_row["learning_rate"]))

    if "Features" in df_top3.columns:
        mlflow.log_param(f"{model_prefix}_best_features", str(best_row["Features"]))

    # Best model metrics
    cv_col = "CV_accuracy" if "CV_accuracy" in df_top3.columns else "CV_Accuracy"

    if cv_col in df_top3.columns:
        mlflow.log_metric(f"{model_prefix}_best_CV_accuracy", float(best_row[cv_col]))

    if "Test_Accuracy" in df_top3.columns:
        mlflow.log_metric(f"{model_prefix}_best_Test_Accuracy", float(best_row["Test_Accuracy"]))

    if "Sensitivity" in df_top3.columns:
        mlflow.log_metric(f"{model_prefix}_best_Sensitivity", float(best_row["Sensitivity"]))

    if "Specificity" in df_top3.columns:
        mlflow.log_metric(f"{model_prefix}_best_Specificity", float(best_row["Specificity"]))

    if "F1" in df_top3.columns:
        mlflow.log_metric(f"{model_prefix}_best_F1", float(best_row["F1"]))

    if "MCC" in df_top3.columns:
        mlflow.log_metric(f"{model_prefix}_best_MCC", float(best_row["MCC"]))

    # Log each Top3 model
    for idx, row in df_top3.iterrows():
        rank = idx + 1

        mlflow.log_param(f"{model_prefix}_top{rank}_model_name", str(row.get("Name_Model", "")))
        mlflow.log_param(f"{model_prefix}_top{rank}_num_features", int(row.get("#Features", 0)))

        if cv_col in df_top3.columns:
            mlflow.log_metric(f"{model_prefix}_top{rank}_CV_accuracy", float(row[cv_col]))

        if "Test_Accuracy" in df_top3.columns:
            mlflow.log_metric(f"{model_prefix}_top{rank}_Test_Accuracy", float(row["Test_Accuracy"]))

        if "Sensitivity" in df_top3.columns:
            mlflow.log_metric(f"{model_prefix}_top{rank}_Sensitivity", float(row["Sensitivity"]))

        if "Specificity" in df_top3.columns:
            mlflow.log_metric(f"{model_prefix}_top{rank}_Specificity", float(row["Specificity"]))


def log_artifacts(files_dict, artifact_folder):
    for artifact_name, file_path in files_dict.items():
        path = Path(file_path)

        if path.exists():
            mlflow.log_artifact(str(path), artifact_path=artifact_folder)
            print(f"Logged artifact: {file_path}")
        else:
            print(f"Skipped missing artifact: {file_path}")


def log_shap_images(shap_dir, artifact_folder):
    if shap_dir.exists():
        for image_path in shap_dir.glob("*.png"):
            mlflow.log_artifact(str(image_path), artifact_path=artifact_folder)
            print(f"Logged SHAP plot: {image_path}")
    else:
        print(f"SHAP folder not found: {shap_dir}")


# ============================================================
# 4) READ TOP3 FILES
# ============================================================

svm_top3 = read_top3_file(files["svm_scores_rbf_top3"])
xgb_top3 = read_top3_file(files["xgb_scores_top3"])


# ============================================================
# 5) LOG MLFLOW RUN
# ============================================================

with mlflow.start_run(run_name="Young_Old_2024_SVM_RBF_XGBoost_Top3_SHAP"):

    # General info
    mlflow.log_param("dataset_name", "Young vs Older 2024")
    mlflow.log_param("experiment_name", "Young_and_Old")
    mlflow.log_param("group_mapping", "Young=0, Older=1")
    mlflow.log_param("feature_selection_method", "SFS")
    mlflow.log_param("tracked_models", "SVM_RBF + XGBoost")
    mlflow.log_param("phase", "Phase_4_4_SHAP")

    # Pipeline notes
    mlflow.log_param("SVM_model_type", "SVM_RBF")
    mlflow.log_param("XGBoost_model_type", "XGBoost")

    # Log SVM Top3
    log_top3_metrics(svm_top3, model_prefix="svm")

    # Log XGBoost Top3
    log_top3_metrics(xgb_top3, model_prefix="xgboost")

    # Log files
    log_artifacts(files, artifact_folder="young_old_files")

    # Log SHAP plots
    log_shap_images(svm_shap_dir, artifact_folder="svm_shap_plots")
    log_shap_images(xgb_shap_dir, artifact_folder="xgboost_shap_plots")

print("Done. Young vs Older MLflow run logged successfully.")