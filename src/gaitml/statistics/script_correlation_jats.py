import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
from src.gaitml.statistics.get_correlation import get_correlation
from src.gaitml.statistics.biomodalitycoeff import bimodality_coeff


def multivars_t_test(feat_names, tf_group, data_cols, group_names):
    """
    Perform t-test and Mann-Whitney U test for each feature between two groups.
    Returns a DataFrame with results.
    """
    records = []
    for idx, varname in enumerate(feat_names):
        data1 = data_cols[tf_group[:, 0], idx]
        data2 = data_cols[tf_group[:, 1], idx]
        _, pval_ttest = ttest_ind(data1, data2, nan_policy='omit')
        _, pval_mannwhitney = mannwhitneyu(data1, data2, alternative='two-sided')
        records.append([varname, pval_ttest, pval_mannwhitney])
    df = pd.DataFrame(records, columns=['Variable Name', 'T-test pval', 'MannWhitney pval'])
    df.to_excel('Multivars_t-tests.xlsx', index=False)
    return df


def correlation_rank(feat_names, tf_group, tabledata):
    """
    Run correlation ranking for each feature and return a report and correlated features list.
    """
    threshold = 0.85
    report = []
    features_corr = []
    del_ft_matrix = []
    for varname in feat_names:
        report_ranks, _, _, features_corr, del_ft = get_correlation(tabledata, varname, threshold, features_corr)
        report.append(report_ranks)
        del_ft_matrix.append(del_ft)
    # Flatten and deduplicate correlated features
    del_ft = list({item for sublist in del_ft_matrix for item in sublist[1:] if item})
    features_corr = list(set(features_corr))
    pd.DataFrame(report).to_excel('Multivars_Corrrank.xlsx', index=False)
    pd.DataFrame(del_ft, columns=['Highly correlated features']).to_excel('Highly correlated features.xlsx', index=False)
    return report, features_corr


def multivar_2groups(feat_names, tf_group, data_cols, group_names):
    """
    Compute multivariate statistics for two groups and save to Excel.
    """
    report_header = [
        'Variable Name', 'Group', 'Mean', 'Std Deviation', 'Median',
        'Biomodality coefficient', 'Normality p-value', 'Normality W-value',
        'T-test pval', 'MannWhitney pval',
    ] + [f'Hist_bin{i+1}' for i in range(10)]
    records = []
    bins = np.linspace(0, 100, 11)
    for idx, varname in enumerate(feat_names):
        data1 = data_cols[tf_group[:, 0], idx]
        data2 = data_cols[tf_group[:, 1], idx]
        # Group 1
        mean1, std1, med1 = np.mean(data1), np.std(data1), np.median(data1)
        bf1, bc1 = bimodality_coeff(data1)
        stat1, pval_norm1 = shapiro(data1)
        hist1, _ = np.histogram(data1, bins=bins)
        # Group 2
        mean2, std2, med2 = np.mean(data2), np.std(data2), np.median(data2)
        bf2, bc2 = bimodality_coeff(data2)
        stat2, pval_norm2 = shapiro(data2)
        hist2, _ = np.histogram(data2, bins=bins)
        # Between group
        _, pval_ttest = ttest_ind(data1, data2, nan_policy='omit')
        _, pval_mannwhitney = mannwhitneyu(data1, data2, alternative='two-sided')
        # Record for group 1
        records.append([
            varname, group_names[0], mean1, std1, med1, bc1[0] if hasattr(bc1, '__iter__') else bc1,
            pval_norm1, stat1, pval_ttest, pval_mannwhitney, *hist1
        ])
        # Record for group 2
        records.append([
            varname, group_names[1], mean2, std2, med2, bc2[0] if hasattr(bc2, '__iter__') else bc2,
            pval_norm2, stat2, pval_ttest, pval_mannwhitney, *hist2
        ])
    df = pd.DataFrame(records, columns=report_header)
    df.to_excel('Multivars_stats_Variant_Normal_57923.xlsx', index=False)
    return df


def main():
    # Load data
    tabledata = pd.read_excel("ALL FEATURES OLDER - Pes Planus and Control JUNE 25.xlsx")
    # Define features
    features_invariant = [
        'Pelv_Angle_Y_MIN_SW_time', 'Pelv_Angle_X_OHS', 'Sha_Foot_Angle_X_HS',
        'Pelv_Angle_Y_ROM', 'Trunk_Angle_X_OTO'
    ]
    modelFeatures = {
        'Normal_57923': [ 'Pelv_Angle_Y_MIN_SW_time', 'Knee_Angle_Y_MIN_SW_time', 'Hip_Angle_X_MAX_SW_time', 'Knee_Angle_X_MAX_SW', 'Hip_Angle_X_MAX_SW' ],
        'Slow_48433': [ 'Cycle_Time', 'Trunk_Angle_X_HS', 'Hip_Angle_Y_ROM', 'Pelv_Angle_X_HS', 'Pelv_Angle_Y_MIN_SW_time' ],
        'VerySlow_31929': [ 'Trunk_Angle_X_ROM', 'Hip_Angle_X_MIN_SW', 'Sha_Foot_Angle_X_HS', 'Knee_Angle_Z_TO', 'Pelv_Angle_Y_MAX_SW' ],
        'Fast_48596': [ 'Hip_Angle_Y_ROM', 'Double_Limb_Support_Time', 'Step_Length', 'Pelv_Angle_Y_MIN_SW_time', 'Pelv_Angle_Z_ROM' ],
        'VeryFast_42632': [ 'Pelv_Angle_Y_MIN_SW_time', 'Double_Limb_Support_Time', 'Pelv_Angle_Y_ROM', 'Knee_Angle_Y_MAX_SW_time', 'Trunk_Angle_Z_MAX_SW_time' ],
    }
    features = modelFeatures['VeryFast_42632']
    filteredTable = tabledata[modelFeatures['Normal_57923']]
    data_cols = filteredTable.values
    feat_names = modelFeatures['Normal_57923']
    properties = tabledata.iloc[:, :4]
    # Define groups
    group = ['Flatfoot', 'Control']
    # Find group indices
    tf_group = np.zeros((properties.shape[0], 2), dtype=bool)
    for i in range(properties.shape[0]):
        tf_group[i, 0] = properties.iloc[i, 2] == group[0]
        tf_group[i, 1] = properties.iloc[i, 2] == group[1]
    # T-tests
    multivars_t_test(feat_names, tf_group, data_cols, group)
    # Correlation rank
    correlation_rank(feat_names, tf_group, tabledata)
    # Multivariate stats
    multivar_2groups(feat_names, tf_group, data_cols, group)

if __name__ == "__main__":
    main() 