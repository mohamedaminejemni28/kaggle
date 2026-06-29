import ast
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# DATASET CONFIGS
# ============================================================

DATASETS = {
    "young_old_2024": {
        "scores_file_top3": "results/young_old_2024_Scores_XGBoost_Top3.xlsx",
        "train_file": "data/processed/young_old_2024_train.xlsx",
        "test_file": "data/processed/young_old_2024_test.xlsx",
        "group_column": "Group",
        "class_mapping": {
            0: "Young",
            1: "Older"
        },
        "output_file": "results/young_old_2024_XGBoost_time_variable_stats.xlsx"
    },

    "autism_2024": {
        "scores_file_top3": "results/autism_2024_Scores_XGBoost_Top3.xlsx",
        "train_file": "data/processed/autism_2024_train.xlsx",
        "test_file": "data/processed/autism_2024_test.xlsx",
        "group_column": "Group",
        "class_mapping": {
            0: "Control",
            1: "Autism"
        },
        "output_file": "results/autism_2024_XGBoost_time_variable_stats.xlsx"
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file_exists(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path


def get_feature_columns(df):
    """
    Extract biomechanical feature columns used in the SVM pipeline.
    """

    start_col = "Pelv_Angle_Y_MAX_SW"
    end_col = "Hip_Angle_Z_OHS"

    if start_col not in df.columns or end_col not in df.columns:
        raise ValueError(
            f"Could not find feature range from '{start_col}' to '{end_col}'.\n"
            f"Available columns are:\n{df.columns.tolist()}"
        )

    return df.loc[:, start_col:end_col].columns.tolist()


def parse_feature_list(value):
    """
    Parse feature values from Excel.

    Possible examples:
        [19 67 18]
        [19, 67, 18]
        ['Feature_A', 'Feature_B']
        Feature_A Feature_B
        Feature_A, Feature_B
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple)):
            return list(parsed)

        return [parsed]

    except Exception:
        pass

    text = text.replace("[", "").replace("]", "")
    text = text.replace("'", "").replace('"', "")
    text = text.replace(",", " ")

    return text.split()


def convert_features_to_names(parsed_features, feature_columns):
    """
    Convert feature indices to feature names when needed.
    If already feature names, keep them.
    """

    feature_names = []

    for item in parsed_features:
        item_str = str(item).strip()

        try:
            idx = int(float(item_str))

            if 0 <= idx < len(feature_columns):
                feature_names.append(feature_columns[idx])
            else:
                print(f"Warning: feature index out of range: {idx}")

        except ValueError:
            feature_names.append(item_str)

    return feature_names


def load_train_test_data(train_file, test_file):
    """
    Load train and test Excel files and combine all sheets.
    """

    train_excel = pd.ExcelFile(train_file)
    test_excel = pd.ExcelFile(test_file)

    combined_dfs = []

    for sheet_name in train_excel.sheet_names:
        df_train = pd.read_excel(train_file, sheet_name=sheet_name)
        df_test = pd.read_excel(test_file, sheet_name=sheet_name)

        df_train["Data_Split"] = "Train"
        df_test["Data_Split"] = "Test"

        df_train["Sheet"] = sheet_name
        df_test["Sheet"] = sheet_name

        combined = pd.concat([df_train, df_test], ignore_index=True)
        combined_dfs.append(combined)

    return pd.concat(combined_dfs, ignore_index=True)


def extract_time_features_from_top3(top3_file, feature_columns):
    """
    Read Top3 SVM file and extract all selected features that contain '_time'.
    """

    excel = pd.ExcelFile(top3_file)

    model_feature_rows = []
    all_time_features = []

    for sheet_name in excel.sheet_names:
        df_scores = pd.read_excel(top3_file, sheet_name=sheet_name)

        if "Features" not in df_scores.columns:
            raise ValueError(
                f"'Features' column not found in sheet '{sheet_name}'.\n"
                f"Available columns: {df_scores.columns.tolist()}"
            )

        for model_rank, row in df_scores.iterrows():
            model_name = row["Name_Model"] if "Name_Model" in df_scores.columns else model_rank + 1

            parsed_features = parse_feature_list(row["Features"])
            feature_names = convert_features_to_names(parsed_features, feature_columns)

            for feature in feature_names:
                is_time_feature = "_time" in feature.lower()

                model_feature_rows.append({
                    "Sheet": sheet_name,
                    "Model_Rank": model_rank + 1,
                    "Name_Model": model_name,
                    "Feature": feature,
                    "Is_Time_Feature": is_time_feature
                })

                if is_time_feature:
                    all_time_features.append(feature)

    unique_time_features = list(dict.fromkeys(all_time_features))

    return unique_time_features, pd.DataFrame(model_feature_rows)


def calculate_stats_by_group(df, features, group_column, class_mapping):
    """
    Calculate descriptive statistics by group.
    """

    rows = []

    for feature in features:
        if feature not in df.columns:
            print(f"Warning: feature not found in dataset: {feature}")
            continue

        for group_value, group_df in df.groupby(group_column):
            values = pd.to_numeric(group_df[feature], errors="coerce").dropna()

            group_label = class_mapping.get(group_value, group_value)
            group_label = class_mapping.get(str(group_value), group_label)

            rows.append({
                "Feature": feature,
                "Group_Value": group_value,
                "Group_Label": group_label,
                "Count": values.count(),
                "Mean": values.mean(),
                "Std": values.std(),
                "Median": values.median(),
                "Min": values.min(),
                "Max": values.max(),
                "Q1": values.quantile(0.25),
                "Q3": values.quantile(0.75),
                "IQR": values.quantile(0.75) - values.quantile(0.25)
            })

    return pd.DataFrame(rows)


def calculate_stats_by_group_and_split(df, features, group_column, class_mapping):
    """
    Calculate descriptive statistics by group and train/test split.
    """

    rows = []

    for feature in features:
        if feature not in df.columns:
            continue

        for (split, group_value), group_df in df.groupby(["Data_Split", group_column]):
            values = pd.to_numeric(group_df[feature], errors="coerce").dropna()

            group_label = class_mapping.get(group_value, group_value)
            group_label = class_mapping.get(str(group_value), group_label)

            rows.append({
                "Feature": feature,
                "Data_Split": split,
                "Group_Value": group_value,
                "Group_Label": group_label,
                "Count": values.count(),
                "Mean": values.mean(),
                "Std": values.std(),
                "Median": values.median(),
                "Min": values.min(),
                "Max": values.max(),
                "Q1": values.quantile(0.25),
                "Q3": values.quantile(0.75),
                "IQR": values.quantile(0.75) - values.quantile(0.25)
            })

    return pd.DataFrame(rows)


def create_raw_time_data(df, features, group_column, class_mapping):
    """
    Create long-format raw values for all time features.
    """

    metadata_columns = [
        col for col in ["Participant", "Index", "Side", "Sheet", "Data_Split", group_column]
        if col in df.columns
    ]

    rows = []

    for feature in features:
        if feature not in df.columns:
            continue

        for _, row in df.iterrows():
            group_value = row[group_column]
            group_label = class_mapping.get(group_value, group_value)
            group_label = class_mapping.get(str(group_value), group_label)

            raw_row = {
                "Feature": feature,
                "Value": row[feature],
                "Group_Value": group_value,
                "Group_Label": group_label
            }

            for col in metadata_columns:
                raw_row[col] = row[col]

            rows.append(raw_row)

    return pd.DataFrame(rows)


def detect_outliers_iqr(df, features, group_column, class_mapping):
    """
    Detect possible outliers using IQR rule inside each group.
    """

    rows = []

    for feature in features:
        if feature not in df.columns:
            continue

        for group_value, group_df in df.groupby(group_column):
            values = pd.to_numeric(group_df[feature], errors="coerce")

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (values < lower_bound) | (values > upper_bound)

            outliers = group_df.loc[mask].copy()

            group_label = class_mapping.get(group_value, group_value)
            group_label = class_mapping.get(str(group_value), group_label)

            for _, row in outliers.iterrows():
                rows.append({
                    "Feature": feature,
                    "Group_Value": group_value,
                    "Group_Label": group_label,
                    "Value": row[feature],
                    "Lower_Bound": lower_bound,
                    "Upper_Bound": upper_bound,
                    "Participant": row["Participant"] if "Participant" in row else "",
                    "Index": row["Index"] if "Index" in row else "",
                    "Side": row["Side"] if "Side" in row else "",
                    "Sheet": row["Sheet"] if "Sheet" in row else "",
                    "Data_Split": row["Data_Split"] if "Data_Split" in row else ""
                })

    return pd.DataFrame(rows)

def create_scatter_plots(df, features, group_column, class_mapping, dataset_name):
    """
    Create scatter plots for each time feature by group.

    X-axis: group
    Y-axis: raw time value
    Each dot: one sample
    """

    output_dir = Path(f"{dataset_name}_SVM_time_variable_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in features:
        if feature not in df.columns:
            print(f"Warning: feature not found for plotting: {feature}")
            continue

        plot_df = df[[group_column, feature, "Data_Split"]].copy()
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
        plot_df = plot_df.dropna(subset=[feature])

        if plot_df.empty:
            print(f"Warning: no valid values for plotting: {feature}")
            continue

        group_values = list(plot_df[group_column].dropna().unique())
        group_values = sorted(group_values)

        group_labels = []
        for group_value in group_values:
            label = class_mapping.get(group_value, group_value)
            label = class_mapping.get(str(group_value), label)
            group_labels.append(str(label))

        plt.figure(figsize=(8, 6))

        for i, group_value in enumerate(group_values):
            values = plot_df.loc[plot_df[group_column] == group_value, feature].values

            # jitter so points do not overlap
            x = np.random.normal(loc=i, scale=0.04, size=len(values))

            plt.scatter(x, values, alpha=0.7)

            # add mean line
            mean_value = np.mean(values)
            plt.hlines(
                y=mean_value,
                xmin=i - 0.2,
                xmax=i + 0.2,
                linewidth=2
            )

        plt.xticks(range(len(group_labels)), group_labels)
        plt.xlabel("Group")
        plt.ylabel(feature)
        plt.title(f"{dataset_name} - {feature}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        safe_feature_name = feature.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_path = output_dir / f"{safe_feature_name}_scatter.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved scatter plot: {output_path}")
def process_dataset(dataset_name, config):
    """
    Process one SVM dataset.
    """

    print("=" * 100)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 100)

    top3_file = config["scores_file_top3"]
    train_file = config["train_file"]
    test_file = config["test_file"]
    group_column = config["group_column"]
    class_mapping = config["class_mapping"]
    output_file = config["output_file"]

    check_file_exists(top3_file)
    check_file_exists(train_file)
    check_file_exists(test_file)

    df_all = load_train_test_data(train_file, test_file)

    first_train_sheet = pd.ExcelFile(train_file).sheet_names[0]
    df_first_train = pd.read_excel(train_file, sheet_name=first_train_sheet)

    feature_columns = get_feature_columns(df_first_train)

    time_features, model_feature_map = extract_time_features_from_top3(
        top3_file=top3_file,
        feature_columns=feature_columns
    )

    print("Time features found:")
    for feature in time_features:
        print(f"- {feature}")

    if len(time_features) == 0:
        print("No time features found in Top3 models.")

    df_time_features = pd.DataFrame({
        "Time_Feature": time_features
    })

    df_stats_group = calculate_stats_by_group(
        df=df_all,
        features=time_features,
        group_column=group_column,
        class_mapping=class_mapping
    )

    df_stats_split = calculate_stats_by_group_and_split(
        df=df_all,
        features=time_features,
        group_column=group_column,
        class_mapping=class_mapping
    )

    df_raw_time = create_raw_time_data(
        df=df_all,
        features=time_features,
        group_column=group_column,
        class_mapping=class_mapping
    )

    df_outliers = detect_outliers_iqr(
        df=df_all,
        features=time_features,
        group_column=group_column,
        class_mapping=class_mapping
    )

    create_scatter_plots(
        df=df_all,
        features=time_features,
        group_column=group_column,
        class_mapping=class_mapping,
        dataset_name=dataset_name
    )
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_time_features.to_excel(writer, sheet_name="Top_Time_Features", index=False)
        model_feature_map.to_excel(writer, sheet_name="Top3_Model_Features", index=False)
        df_stats_group.to_excel(writer, sheet_name="Stats_By_Group", index=False)
        df_stats_split.to_excel(writer, sheet_name="Stats_By_Group_Split", index=False)
        df_raw_time.to_excel(writer, sheet_name="Raw_Time_Data", index=False)
        df_outliers.to_excel(writer, sheet_name="Potential_Outliers", index=False)

    print(f"Done. Output created: {output_file}")







# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for dataset_name, config in DATASETS.items():
        process_dataset(dataset_name, config)

    print("=" * 100)
    print("All SVM time-variable stats completed.")
    print("=" * 100)