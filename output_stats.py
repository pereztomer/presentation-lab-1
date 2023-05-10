import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay


def main():
    # Generate example data
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.5, 0.55, 0.32, 0.35, 0.5, 0.6, 0.7, 0.8, 0.6, 0.9])
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Plot ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    # Calculate confusion matrix
    y_pred = y_scores > 0.5
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    # initialize using the raw 2D confusion matrix
    # and output labels (in our case, it's 0 and 1)
    display = ConfusionMatrixDisplay(conf_mat, display_labels=[0, 1])

    # set the plot title using the axes object
    # ax.set(title='Confusion Matrix for the Diabetes Detection Model')

    # show the plot.
    # Pass the parameter ax to show customizations (ex. title)
    display.plot()
    plt.show()

    # fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    #
    # # initialize using the raw 2D confusion matrix
    # # and output labels (in our case, it's 0 and 1)
    # display = ConfusionMatrixDisplay(test_cm, display_labels=model.classes_)
    #
    # # set the plot title using the axes object
    # ax.set(title='Confusion Matrix for the Diabetes Detection Model')
    #
    # # show the plot.
    # # Pass the parameter ax to show customizations (ex. title)
    # display.plot(ax=ax)
    # plt.show()


if __name__ == '__main__':
    main()
