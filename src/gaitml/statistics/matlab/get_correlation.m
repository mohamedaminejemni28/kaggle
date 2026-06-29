function [report_layout, corrcoeffiecient, feat_names, features_corr, delete_ft] = get_correlation(tabledata, varname, threshold, features_corr)
    data_cols   = tabledata{:, 5:end};
    feat_names  = (tabledata.Properties.VariableNames(5:end))';
    properties  = tabledata(:, 1:4);
    
    %% correlation
    for i=1:length(feat_names)
        for j = 1:length(feat_names)
            if ~(i==j)
                R = corrcoef(data_cols(:,i),data_cols(:,j));
            else
                R = [1, 1; 1, 1];
            end
            corrcoeffiecient(i,j) = R(1,2);
        end
    end
    
    % %% write to file
    % report_correlation = [header_cols', num2cell(corrcoeffiecient)];
    % report_correlation_header = ['Correlation', header_cols];
    % report_correlation = [report_correlation_header; report_correlation];
    % filename = strcat('CorrelationMatrix','.xlsx');
    % writecell(report_correlation,filename,'Sheet',1,'Range','A1')
    
    % find column index
    for i=1:length(feat_names)
        tf(i) = strcmp(feat_names{i},varname);
    end
    
    corrs = corrcoeffiecient(:,tf);
    [B,I] = sort(corrs, 'descend');
    
    filtered_corrs = B(B>threshold );
    filtered_corrs_locs = I(B>threshold);
    header_cols_filtered = feat_names(filtered_corrs_locs);
    filtered_corrs_string=arrayfun(@(x) sprintf('%10.4f',x),filtered_corrs,'un',0);
    report_layout = cell(2, 115);
    report = [header_cols_filtered'; filtered_corrs_string'];
    report_layout(:, 1:size(report,2)) = report;

    features_corr = [features_corr, header_cols_filtered(2:end)'];
    delete_ft = header_cols_filtered';

    % Create iteration for the delete feature list to skip the already identfied
    % features
        % Validate that the list had identfied some features with
        % correlation
    
% if only_names
%     report_layout = report_layout(1,:);
% end
end
