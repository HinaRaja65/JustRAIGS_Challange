import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc


def get_specificity(y_test, y_pred):
    """

    :param y_test:
    :param y_pred:
    :return:
    specificity at 0.95 value of sensitivity
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(fpr, tpr)

    desired_specificity = 0.95
    # Find the index of the threshold that is closest to the desired specificity
    idx = np.argmax(fpr >= (1 - desired_specificity))
    # Get the corresponding threshold
    threshold_at_desired_specificity = round(thresholds[idx], 4)
    # Get the corresponding TPR (sensitivity)
    sensitivity_at_desired_specificity = round(tpr[idx], 4)

    return sensitivity_at_desired_specificity
