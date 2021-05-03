import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score, \
    average_precision_score, precision_score, recall_score, \
    precision_recall_curve, roc_curve, roc_auc_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter


def df_correlation(df, width, height, print_coeff, thresh=0, method='Pearson'):
    """Plot the correlation plot of a dataframe columns

    Parameters
    -------------
    df: dataframe
      The dataframe from which we want to plot columns correlation
    width : int
      the width of the figure
    height : int
      the height of the figure
    print_coeff : bool
      If True, shows the correlation coefficient values in the plot
    thresh : float
      Nothing is plotted if the correlation value is inferior to this threshold in absolute value
    method : str
      defines the method used for correlation processing, default is Pearson's. Takes value in {"pearson"(default),"kendall","spearman" or callable}
    """
    cormat = df.corr(method=method)
    cormat = cormat.where(abs(cormat) >= thresh)
    cormat = cormat.fillna(0)

    mask = np.zeros_like(cormat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(cormat, vmin=-1, vmax=1, cmap="coolwarm", annot=print_coeff, mask=mask);