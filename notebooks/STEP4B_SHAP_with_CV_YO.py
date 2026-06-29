# importing necessary packages
import matplotlib.pyplot as plt  # for making plots / graphs
import pandas as pd              # for reading the .csv file and related operations
import numpy as np               # for working with arrays (multi-dimensional)
from sklearn.svm import SVC
import shap # Use SHAP module to get Model Interpretability.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#LOCATE PATH
path_file = "Variant Speeds Young and Older\Scores_RBF_variant.xlsx"
path_file_features = "Variant Speeds Young and Older"
paramaters_EXCEL = pd.ExcelFile(path_file)

# PATHS DATASETS
path_test = "Variant Speeds Young and Older\Post-processing Speeds Young and Older Test.xlsx"
path_train = "Variant Speeds Young and Older\Post-processing Speeds Young and Older Train.xlsx"
test_EXCEL  = pd.ExcelFile(path_test)
train_EXCEL =  pd.ExcelFile(path_train)

#Identify data models
dfs = []
for sheet in paramaters_EXCEL.sheet_names:
    df = pd.read_excel(paramaters_EXCEL, sheet_name=sheet)
    df['SPEED_MODEL'] = sheet
    dfs.append(df)

models_parameters = pd.concat(dfs, ignore_index=True)

df_SHAP = {}
for i in range(len(models_parameters)):
    # SPEED MODEL
    SHEET_SPEED = models_parameters['SPEED_MODEL'][i]
    CV_SPLIT     = models_parameters['CV_split'][i]
    RANDOM_STATE = models_parameters['Random_State'][i]
    # model parameters
    C_VAL = models_parameters['C_Value'][i]
    R_VAL = models_parameters['Gamma_Value'][i]
    # Selected Feats
    FEATS = models_parameters['Features'][i]
    FEATS = FEATS.strip("[]").split()
    FEATS = np.array(FEATS)
    
    SHAP_NAME = 'SHAP_'+str(models_parameters['Name_Model'][i])
    
    # Read the YO TRAIN dataset
    df_YO_train = pd.read_excel(train_EXCEL, sheet_name=SHEET_SPEED)
    print("df.shape = ", df_YO_train)
    
    df_YO_train["Group"] = df_YO_train["Group"].replace({'Young': 0, 'Older': 1})
    
    Y_YO_train = df_YO_train.loc[:, "Group"].values
    print("y.shape = ", Y_YO_train.shape)
    print("y = ", Y_YO_train)
    
    segments_YO_train = {
        'All_Data' : df_YO_train.loc[:, 'Pelv_Angle_Y_MAX_SW' : 'Hip_Angle_Z_OHS']
    }
    
    # read the YO TEST dataset
    df_YO_test = pd.read_excel(test_EXCEL, sheet_name=SHEET_SPEED)
    print("df.shape = ", df_YO_test)
    
    df_YO_test["Group"] = df_YO_test["Group"].replace({'Young': 0, 'Older': 1})
    
    Y_YO_test = df_YO_test.loc[:, "Group"].values
    print("y.shape = ", Y_YO_test.shape)
    print("y = ", Y_YO_test)
    
    segments_YO_test = {
        'AllF' : df_YO_test.loc[:, 'Pelv_Angle_Y_MAX_SW' : 'Hip_Angle_Z_OHS']
    }
    
    print(segments_YO_train["All_Data"].shape)
    print(segments_YO_test["AllF"].shape)
    
    # merge the AM and NM dataset
    df_All_set = pd.concat([df_YO_train, df_YO_test], ignore_index=True)
    print("df_AMNM.shape = ", df_All_set.shape)
    
    Y_YO_All = df_All_set.loc[:, "Group"].values
    print("Y_AMNM = ", Y_YO_All)
    
    segments_ALL = {
        'AllF' : df_All_set.loc[:, 'Pelv_Angle_Y_MAX_SW' : 'Hip_Angle_Z_OHS'],
    }
    
    # get the names of the feature subset selected using the Feature Selection algorithm.
    sfs_feature_names = np.array([int(i) for i in FEATS])
    print("Number of Features selected: ", len(sfs_feature_names))
    
    df_segment_train = segments_YO_train["All_Data"].iloc[:, sfs_feature_names]
    df_segment_test = segments_YO_test["AllF"].iloc[:, sfs_feature_names]
    df_segment_all = segments_ALL["AllF"].iloc[:, sfs_feature_names]
    
    cols = df_segment_all.columns.values.tolist()
    print(len(cols))
    print(cols)
    
    #SHAP on TRAIN+TEST Dataset
    X_train = df_segment_train
    Y_train = Y_YO_train
    X_test  = df_segment_all
    Y_test  = Y_YO_All
    X_YO_test   = df_segment_test

    # the training dataset
    print("Training Data Shape = ", X_train.shape)
    print("Training Data Values Shape = ", Y_train.shape)
    # the testing dataset
    print("Testing Data Shape = ", X_test.shape)
    print("Testing Data Values Shape = ", Y_test.shape)
    
    # applying the scaler
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_YO_test = sc.transform(X_YO_test)
    
    # defining the SVM model
    svm = SVC(kernel = 'rbf', C = C_VAL, gamma = R_VAL)
    svm.fit(X_train, Y_train)
    print("Only TEST Accuracy = ", svm.score(X = X_YO_test, y = Y_YO_test))
    print("TRAIN+TEST Accuracy = ", svm.score(X = X_test, y = Y_test))
    
    # Use SHAP to explain predictions
    svm_explainer = shap.KernelExplainer(svm.predict, X_test, feature_names=cols)
    shap_values = svm_explainer.shap_values(X_test)
    
    plt.figure()
    shap.summary_plot(shap_values, feature_names=cols, plot_type = "bar", max_display= 50, show=False)
    plt.savefig(SHAP_NAME+"AMNM.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    vals = np.array(shap_values)
    shap_values = vals
    
    df_SHAP_Vals = pd.DataFrame(shap_values, columns = cols)
    df_SHAP_Vals = pd.concat([df_All_set[['Participant', 'Index', 'Side', 'Group']], df_SHAP_Vals], axis = 1)
    
    df_SHAP_Vals['SUM'] = df_SHAP_Vals[cols].sum(axis=1)
    
    min_cols = df_SHAP_Vals[cols].idxmin(axis=1)
    max_cols = df_SHAP_Vals[cols].idxmax(axis=1)
    df_SHAP_Vals['Most Contributing Feat'] = np.where(df_SHAP_Vals['Group'] == 0, min_cols, max_cols)
    df_SHAP_Vals['Most Contributing Val'] = df_SHAP_Vals.apply(lambda row: row[row['Most Contributing Feat']], axis = 1)
    
    shap_name = SHEET_SPEED + '_' + str(models_parameters['Name_Model'][i])
    df_SHAP[shap_name] = df_SHAP_Vals

file_name = 'Variant Speeds Young and Older\SHAP_analysis.xlsx'
# Save all sheets to a new Excel file
with pd.ExcelWriter(file_name) as writer:
    for sheet_name, df_SHAP in df_SHAP.items():
        df_SHAP.to_excel(writer, sheet_name=sheet_name, index=False)