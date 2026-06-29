% ***************************
% Date: 15th January 2025
% This code gets the correlation features focused only on 
% Joint Angles + Temporal Spatial Features
%
% Created By: José Rodrigo Quintero Valdez
%***************************

%% Cleaning workspace
clear; close all; clc;

%% Read Invariant table data
tabledata = readtable("ALL FEATURES OLDER - Pes Planus and Control JUNE 25.xlsx");

%% Variant set features model chosen
% Define the list of features of invariant model you want to extract
features_invariant = {'Pelv_Angle_Y_MIN_SW_time', ...
            'Pelv_Angle_X_OHS', ...
            'Sha_Foot_Angle_X_HS', ...
            'Pelv_Angle_Y_ROM', ...
            'Trunk_Angle_X_OTO'};

% List of features of variant models
modelFeatures.Normal_57923  = { 'Pelv_Angle_Y_MIN_SW_time', 'Knee_Angle_Y_MIN_SW_time', 'Hip_Angle_X_MAX_SW_time', 'Knee_Angle_X_MAX_SW', 'Hip_Angle_X_MAX_SW' };
modelFeatures.Slow_48433      = { 'Cycle_Time', 'Trunk_Angle_X_HS', 'Hip_Angle_Y_ROM', 'Pelv_Angle_X_HS', 'Pelv_Angle_Y_MIN_SW_time' };
modelFeatures.VerySlow_31929  = { 'Trunk_Angle_X_ROM', 'Hip_Angle_X_MIN_SW', 'Sha_Foot_Angle_X_HS', 'Knee_Angle_Z_TO', 'Pelv_Angle_Y_MAX_SW' };
modelFeatures.Fast_48596      = { 'Hip_Angle_Y_ROM', 'Double_Limb_Support_Time', 'Step_Length', 'Pelv_Angle_Y_MIN_SW_time', 'Pelv_Angle_Z_ROM' };
modelFeatures.VeryFast_42632  = { 'Pelv_Angle_Y_MIN_SW_time', 'Double_Limb_Support_Time', 'Pelv_Angle_Y_ROM', 'Knee_Angle_Y_MAX_SW_time', 'Trunk_Angle_Z_MAX_SW_time' };

features = modelFeatures.VeryFast_42632;

% Filter the table to include only these columns
filteredTable = tabledata(:, modelFeatures.Normal_57923);

%% Keep properties, index, and numeric data
%data_cols   = tabledata{:, 5:end};
data_cols = filteredTable{:,:};
%feat_names  = (tabledata.Properties.VariableNames(5:end))';
feat_names = features;
properties  = tabledata(:, 1:4);



%% Define groups
group = {'Flatfoot', 'Control'};

%% Define reports header
% Initialize report
report_header = [{'Variable Name'}, {'T-test pval'}, {'MannWhitney pval'}];
report = report_header;

%% Find group indices
tf_group = false(size(properties, 1), 2);
for i = 1:size(properties, 1)
    tf_group(i, 1) = strcmp(properties{i, 3}, group{1});
    tf_group(i, 2) = strcmp(properties{i, 3}, group{2});
end

%% Obtain T-Test
T_test_report = Multivars_t_test(feat_names, tf_group, data_cols);

%% Get_correlation_rank
[correlation_rank_report, features_corr_list] = correlation_rank(feat_names, tf_group, tabledata);

%% Obtain_multivar_2groups
[full_report] = multivar_2groups(feat_names, tf_group, data_cols, group);

%% Function to create report of Multivars_t-test
function [report] = Multivars_t_test(feat_names, tf_group, data_cols)
    report = [];

    % Perform statistical tests and generate report
    for idx_var = 1:length(feat_names)
        % Get the Feature name iterating from the Feature List
        varname = feat_names{idx_var};
    
        % Get the column index for the current variable
        col_idx = strcmp(feat_names, varname);
    
        % Filter data for each group
        data_cols_flatfoot = data_cols(tf_group(:, 1), col_idx);
        data_cols_control = data_cols(tf_group(:, 2), col_idx);
    
        % Perform t-test
        [~, pval_ttest] = ttest2(data_cols_flatfoot, data_cols_control);
    
        % Perform Mann-Whitney U test
        pval_mannwhitney = ranksum(data_cols_flatfoot, data_cols_control);
    
        % Add results to the report
        report = [report; {varname, pval_ttest, pval_mannwhitney}];
    end
    
    % Write the report to an Excel file
    filename = 'Multivars_t-tests.xlsx';
    writecell(report, filename, 'Sheet', 1, 'Range', 'A1');
end

%% Function to create report of Multivars_Corrank
function [report, features_corr] = correlation_rank(feat_names, tf_group, tabledata)
    % Inicialate variables for identify highly correlations
    threshold = 0.85;
    report        = [];
    features_corr = [];
    del_ft_matrix = cell(114, 114);

    for idx_var = 1:length(feat_names)
        % Identify feature name from the list
        varname = feat_names{idx_var};

        % Get correlations
        [report_ranks, corrcoeffiecient, header_cols, features_corr, del_ft] = get_correlation(tabledata, varname, threshold, features_corr);

        report = [report; report_ranks];
        del_ft_matrix(idx_var, 1:length(del_ft)) = del_ft;
    end

    % Obtain features delete list with highly correlation
    del_ft = [];
    clean_ft_names = feat_names;

    for i = 1:length(del_ft_matrix)
        % Identify the highly correlation for every feature
        for j = 2:length(del_ft_matrix)
            if iscellstr(del_ft_matrix(i,j))  
                % Create the fistly appeared correlated feature list
                del_ft = [del_ft; del_ft_matrix(i,j)];
                % Find the index to evoid the highly correlated feature
                index_tf = find(strcmpi(clean_ft_names, del_ft_matrix(i,j)));
                % Convert that feature to [] on the high correlated feature matrix
                del_ft_matrix(index_tf,:) = {[]};
            end 
        end
    end
    del_ft = unique(del_ft);
    features_corr = features_corr';

    %Save Corrank_report
    filename = strcat('Multivars','_Corrrank','.xlsx');
    writecell(report,filename,'Sheet',1,'Range','A1')

    %Save unique highly correlated features
    filename = strcat('Highly correlated features','.xlsx');
    writecell(del_ft,filename,'Sheet',1,'Range','A1')
end

%% Function multivar for two groups
function [report] = multivar_2groups(feat_names, tf_group, data_cols, group)
    
    report_header_final = [{'Variable Name'}, {'Group'},  {'Mean'},   {'Std Deviation'},  {'Median'},...
                       {'Biomodality coefficient'},   {'Normality p-value'},          {'Normality W-value'},...
                       {'T-test pval'},   {'MannWhitney pval'},   {'Hist_bin1'},      {'Hist_bin2'},...
                       {'Hist_bin3'},     {'Hist_bin4'},          {'Hist_bin5'},      {'Hist_bin6'},...
                       {'Hist_bin7'},     {'Hist_bin8'},          {'Hist_bin9'},      {'Hist_bin10'}];

    report = report_header_final;

    %Create a for loop to iterate between the features
    for idx_ft = 1:length(feat_names)
        % Get the variable name
        varname = feat_names(idx_ft);
        
        % Get the data from the feauture column
        % Group 1: Flatfoot
        data_cols_group1 = data_cols(tf_group(:,1),idx_ft);
        % Group 2: Control
        data_cols_group2 = data_cols(tf_group(:,2),idx_ft);

        average(1)  = mean(data_cols_group1,1);     % Average group 1
        standdev(1) = std(data_cols_group1,1);      % Standard deviation group 1
        medval(1) = median(data_cols_group1,1);     % Median group 1
        average(2) = mean(data_cols_group2,1);      % Average group 2
        standdev(2) = std(data_cols_group2,1);      % Standard deviation group 2
        medval(2) = median(data_cols_group2,1);     % Median group 2
        
        % Calculate Shapiro-Wilk parametric hypothesis test of composite
        % normality for group 1
        [h,p(1), w(1)] = swtest(data_cols_group1);
        
        % Calculate histogram bin counts group 1
        [hist_10vals(:,1), n] = histcounts(data_cols_group1,[0, 10,20,30,40,50,60,70,80,90,101]);

        % Bimodality Coefficient Calculation group 1
        [BF, BC(1)] = bimodalitycoeff(data_cols_group1);
        
        % Calculate Shapiro-Wilk parametric hypothesis test of composite
        % normality for group 2
        [h,p(2), w(2)] = swtest(data_cols_group2);

        % Calculate histogram bin counts group 2
        [hist_10vals(:,2), n] = histcounts(data_cols_group2,[0, 10,20,30,40,50,60,70,80,90,101]);

        % Bimodality Coefficient Calculation group 2
        [BF, BC(2)] = bimodalitycoeff(data_cols_group2);
        
        % Between Group, p value
        [h,pval]    = ttest2(data_cols_group1, data_cols_group2);
        [pval_np,h] = ranksum(data_cols_group1,data_cols_group2);
        
        report1 = [varname, group(1), num2cell(average(1)'), num2cell(standdev(1)'),...
            num2cell(medval(1)'), num2cell(BC(1)'),num2cell(p(1)'),num2cell(w(1)'),...
            num2cell(pval'), num2cell(pval_np'), num2cell(hist_10vals(:,1)')];
        report2 = [varname, group(2), num2cell(average(2)'), num2cell(standdev(2)'),...
            num2cell(medval(2)'), num2cell(BC(2)'),num2cell(p(2)'),num2cell(w(2)'),...
            num2cell(pval'), num2cell(pval_np'), num2cell(hist_10vals(:,2)')];
        report = [report;report1;report2];
    end

    %filename = strcat('Multivars','_stats','.xlsx');
    filename = strcat('Multivars','_stats','_Variant_Normal_57923','.xlsx');
    writecell(report,filename,'Sheet',1,'Range','A1')
end