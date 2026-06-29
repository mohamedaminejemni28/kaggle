import numpy as np
import pandas as pd

def get_correlation(tabledata, varname, threshold, features_corr=None):
    """
    Compute correlation matrix, identify highly correlated features, and prepare a report.

    Parameters
    ----------
    tabledata : pd.DataFrame
        DataFrame with at least 5 columns; features start from column 5.
    varname : str
        The variable name to compare correlations against.
    threshold : float
        Correlation threshold for flagging features.
    features_corr : list, optional
        List of already found correlated features (default: None).

    Returns
    -------
    report_layout : list of lists
        Layout for reporting highly correlated features and their coefficients.
    corrcoeffiecient : np.ndarray
        Correlation coefficient matrix.
    feat_names : list
        List of feature names.
    features_corr : list
        Updated list of correlated features.
    delete_ft : list
        List of features to consider for deletion (highly correlated).
    """
    if features_corr is None:
        features_corr = []
    # Features start from column 5 (index 4)
    data_cols = tabledata.iloc[:, 4:].values
    feat_names = list(tabledata.columns[4:])
    properties = tabledata.iloc[:, :4]

    # Compute correlation matrix
    n_feats = len(feat_names)
    corrcoeffiecient = np.ones((n_feats, n_feats))
    for i in range(n_feats):
        for j in range(n_feats):
            if i != j:
                corr = np.corrcoef(data_cols[:, i], data_cols[:, j])[0, 1]
                corrcoeffiecient[i, j] = corr

    # Find column index for varname
    try:
        tf = feat_names.index(varname)
    except ValueError:
        raise ValueError(f"Variable name '{varname}' not found in feature names.")

    corrs = corrcoeffiecient[:, tf]
    B = np.sort(corrs)[::-1]
    I = np.argsort(corrs)[::-1]

    filtered_corrs = B[B > threshold]
    filtered_corrs_locs = I[B > threshold]
    header_cols_filtered = [feat_names[idx] for idx in filtered_corrs_locs]
    filtered_corrs_string = [f"{x:10.4f}" for x in filtered_corrs]

    # Prepare report layout (2 rows: feature names, correlation values)
    report_layout = [header_cols_filtered, filtered_corrs_string]
    # Pad to 115 columns as in MATLAB (optional, for compatibility)
    max_cols = 115
    for row in report_layout:
        row += [''] * (max_cols - len(row))

    # Update features_corr and delete_ft
    features_corr = features_corr + header_cols_filtered[1:]
    delete_ft = header_cols_filtered

    return report_layout, corrcoeffiecient, feat_names, features_corr, delete_ft


if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.random.rand(100),
        'D': np.random.rand(100),
        'E': np.random.rand(100),
        'F': np.random.rand(100),
    })
    # Add dummy columns to simulate properties
    for col in ['P1', 'P2', 'P3', 'P4']:
        df.insert(0, col, np.random.randint(0, 2, 100))
    report, corrmat, names, feats_corr, delete_ft = get_correlation(df, 'E', 0.7)
    print("Report Layout:", report)
    print("Features to delete:", delete_ft) 