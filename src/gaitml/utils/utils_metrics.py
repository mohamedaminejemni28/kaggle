# -*- coding: utf-8 -*-
"""
Utils Metrics Module

Created on Tue Jul  6 13:09:42 2021
@author: Karansinh Padhiar, roqui

This module provides functions for calculating various classification metrics.
It includes two types of functions:
1. Functions that work with y_true and y_pred arrays (using sklearn)
2. Functions that work with confusion matrix components (tn, fp, fn, tp)

Reference: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
"""

# importing necessary packages
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import matthews_corrcoef
from numpy import sqrt
import numpy as np
# =============================================================================
# FUNCTIONS THAT WORK WITH y_true AND y_pred ARRAYS (using sklearn)
# =============================================================================

def get_sensitivity(y_true, y_pred):
    """Calculate sensitivity (True Positive Rate) using sklearn"""
    return recall_score(y_true=y_true, y_pred=y_pred, pos_label=1)

def get_specificity(y_true, y_pred):
    """Calculate specificity (True Negative Rate) using sklearn"""
    return recall_score(y_true=y_true, y_pred=y_pred, pos_label=0)

def get_NPV(y_true, y_pred):
    """Calculate Negative Predictive Value using sklearn"""
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    if tn + fn == 0:
        return float(tn) / 1
    return float(tn) / (tn + fn)

def get_PPV(y_true, y_pred):
    """Calculate Positive Predictive Value (Precision) using sklearn"""
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    if tp + fp == 0:
        return float(tp) / 1
    return float(tp) / (tp + fp)

def get_PLR(y_true, y_pred):
    """Calculate Positive Likelihood Ratio using sklearn"""
    if (1 - get_specificity(y_true, y_pred)) == 0:
        return get_sensitivity(y_true, y_pred) / 1
    return get_sensitivity(y_true, y_pred) / (1 - get_specificity(y_true, y_pred))

def get_MCC(y_true, y_pred):
    """Calculate Matthews Correlation Coefficient using sklearn"""
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return float(numerator) / 1
    return float(numerator) / denominator

# =============================================================================
# FUNCTIONS THAT WORK WITH CONFUSION MATRIX COMPONENTS (tn, fp, fn, tp)
# =============================================================================

def get_MCC_from_cm(tn, fp, fn, tp):
    """Calculate Matthews Correlation Coefficient from confusion matrix components"""
    numerator = (tp * tn) - (fp * fn)
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return float(numerator) / 1
    return float(numerator) / denominator

def get_specificity_from_cm(tn, fp, fn, tp):
    """Calculate specificity (True Negative Rate) from confusion matrix components"""
    if tn + fp == 0:
        return float(tn) / 1
    return float(tn) / (tn + fp)

def get_sensitivity_from_cm(tn, fp, fn, tp):
    """Calculate sensitivity (True Positive Rate) from confusion matrix components"""
    if tp + fn == 0:
        return float(tp) / 1
    return float(tp) / (tp + fn)

def get_f1_from_cm(tp, fp, fn, tn):
    """Calculate F1 score from confusion matrix components"""
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    # Calculate F1 score
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score

def get_likelihood_ratios_from_cm(tp, fp, fn, tn):
    """Calculate negative likelihood ratio (LR-) from confusion matrix components"""
    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Calculate LR-
    lr_minus = (1 - sensitivity) / specificity if specificity != 0 else float('inf')

    return lr_minus

def get_ppv_from_cm(tp, fp):
    """Calculate Positive Predictive Value (Precision) from confusion matrix components"""
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
    return ppv

def get_npv_from_cm(tn, fn):
    """Calculate Negative Predictive Value from confusion matrix components"""
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    return npv

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_confusion_matrix(matrix_str):
    """Parse confusion matrix string to numpy array"""
    matrix = matrix_str.replace('\n ', ' ').replace('  ', ',').replace('[ ','[').replace(' [',',[').replace(' ',',')
    return np.array(eval(matrix))

# =============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

# Aliases for confusion matrix component functions (for backward compatibility)
get_MCC_cm = get_MCC_from_cm
get_specificity_cm = get_specificity_from_cm
get_sensitivity_cm = get_sensitivity_from_cm
get_f1_cm = get_f1_from_cm
get_likelihood_ratios_cm = get_likelihood_ratios_from_cm
get_ppv_cm = get_ppv_from_cm
get_npv_cm = get_npv_from_cm 