import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay

import numpy as np
import pandas as pd
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies,model_name):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# https://github.com/ast0414/CSE6250BDH-LAB-DL/blob/master/2_CNN.ipynb
	fig, ax = plt.subplots(2,figsize=(10, 10))
	pd.Series(train_losses).plot(title='Loss', label='Train',ax=ax[0])
	pd.Series(valid_losses).plot(title='Loss', label='Validation',ax=ax[0])

	pd.Series(train_accuracies).plot(title='Loss', label='Train',ax=ax[1])
	pd.Series(valid_accuracies).plot(title='Loss', label='Validation',ax=ax[1])

	ax[0].set_xlabel("epoch")
	ax[1].set_xlabel("epoch")

	ax[0].set_ylabel("Loss")
	ax[1].set_ylabel("Accuracy")

	ax[0].legend(loc='lower left')
	ax[1].legend(loc='lower left')
	plt.tight_layout()
	plt.savefig(f'learning_curves_{model_name}.png')

def plot_confusion_matrix(results, class_names,model_name):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
	
	# results.extend(list(zip(y_true, y_pred)))
	np_results=np.array(results)
	y_true,y_pred=np_results[:,0],np_results[:,1]

	# normalize{‘true’, ‘pred’, ‘all’}, default=None
	# Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. 
	# If None, confusion matrix will not be normalized.

	# where each element is fraction of the predicted class
	# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
	# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
	# labels=["ant", "bird", "cat"]
	confusion_matrix_result=confusion_matrix(y_true,y_pred,normalize='pred')
	# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
	confusion_matrix_plot=ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_result,display_labels=class_names)
	fig, ax = plt.subplots(figsize=(10, 10))
	confusion_matrix_plot.plot(cmap=plt.cm.Blues,ax=ax)
	plt.tight_layout()
	plt.savefig(f'confusion_matrix_{model_name}_plot.png')
	# import pdb;pdb.set_trace()

