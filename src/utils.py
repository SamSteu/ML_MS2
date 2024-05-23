import numpy as np 
import matplotlib.pyplot as plt
import os

# Generaly utilies
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)

def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (array): of shape (N,D)
    Returns:
        (array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]),data], axis=1)
    return data

def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.
    
    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds

def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.
    
    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)

def mse_fn(pred,gt):
    '''
        Mean Squared Error
        Arguments:
            pred: NxD prediction matrix
            gt: NxD groundtruth values for each predictions
        Returns:
            returns the computed loss

    '''
    loss = (pred-gt)**2
    loss = np.mean(loss)
    return loss


# Plotting function
#############

def visualize_histogram(labels_train, labels_test):
    class_names = ['0 Top/T-Shirt', '1 Trouser', '2 Pullover', '3 Dress', '4 Coat', '5 Sandal', '6 Shirt', '7 Sneaker', '8 Bag', '9 Ankle Boot']
    n_classes = get_n_classes(labels_train)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Number of items in each clothing category')

    # train data histogram
    axs[0].bar(np.arange(n_classes), np.bincount(labels_train))
    axs[0].set_title('Training data')
    axs[0].set_ylabel('Count')
    axs[0].set_xticks(np.arange(n_classes))
    axs[0].set_xticklabels(class_names, rotation=45, ha="right")
    axs[0].grid(True)
    axs[0].set_axisbelow(True)

    # test data histogram
    axs[1].bar(np.arange(n_classes), np.bincount(labels_test))
    axs[1].set_title('Validation set data')
    axs[1].set_ylabel('Count')
    axs[1].set_xticks(np.arange(n_classes))
    axs[1].set_xticklabels(class_names, rotation=45, ha="right")
    axs[1].grid(True)
    axs[1].set_axisbelow(True)

    fig.subplots_adjust(hspace=0.5)
    plt.show()




# Générer un nom de fichier unique
def get_unique_filename(directory, base_filename, extension):
    counter = 1
    filename = f"{base_filename}{extension}"
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base_filename}_{counter}{extension}"
        counter += 1
    return os.path.join(directory, filename)



def roc_curve(y_true, y_score, titre, acc_train, macrof1_train, acc_test, macrof1_test, pos_label=1):
    """
    Compute Receiver operating characteristic (ROC) curve.
    
    Parameters:
    y_true (list or array-like): True binary labels.
    y_score (list or array-like): Target scores, can either be probability estimates of the positive class,
                                  confidence values, or binary decisions.
    pos_label (int): Label considered as positive and others are considered negative.
    
    Returns:
    fpr (array): False positive rates.
    tpr (array): True positive rates.
    thresholds (array): Thresholds on the decision function used to compute FPR and TPR.
    """
    # Convert input to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Sort scores and corresponding true values
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]
    
    # Unique threshold values
    thresholds = np.unique(y_score)
    thresholds = np.append(thresholds, thresholds[-1] + 1)  # Ensure the last threshold is unique and larger
    tpr = np.zeros_like(thresholds, dtype=float)
    fpr = np.zeros_like(thresholds, dtype=float)
    
    # Calculate TPR and FPR at each threshold
    for i, threshold in enumerate(thresholds):
        y_pred = y_score >= threshold
        tp = np.sum((y_pred == 1) & (y_true == pos_label))
        fp = np.sum((y_pred == 1) & (y_true != pos_label))
        fn = np.sum((y_pred == 0) & (y_true == pos_label))
        tn = np.sum((y_pred == 0) & (y_true != pos_label))
        
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0


    # Plotting the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker='o', linestyle='-', color='b')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC curve for :\n" + 
            titre +"\n" +
            f"Train set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}\n" +
            f"Validation set:  accuracy = {acc_test:.3f}% - F1-score = {macrof1_test:.6f}") 
    plt.grid()
    
    base_filename = "ROC_" + titre
    extension = ".png"
    output_dir = "graph_scores"
    unique_filename = get_unique_filename(output_dir, base_filename, extension)

    plt.savefig(unique_filename)
    
    return fpr, tpr, thresholds


def plot_epoch_score(epoch_acc, epoch_f1, titre, acc_train, macrof1_train, acc_test, macrof1_test):
    print("Scores during training phase")
    n = len(epoch_acc)
    plt.figure(figsize=(11,8))
    plt.title("Scores during training phase for each epoch :\n" + 
            titre +"\n" +
            f"Train set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}\n" +
            f"Validation set:  accuracy = {acc_test:.3f}% - F1-score = {macrof1_test:.6f}") 
    plt.ylabel("Score [%]")
    plt.xlabel("Epoch number")

    plt.plot(np.arange(n), epoch_acc, label = "Accuracy")
    plt.plot(np.arange(n), epoch_f1, label = "F1 score")
    #plt.xticks(np.arange(n))
    plt.legend()
    
    
    base_filename = titre
    extension = ".png"
    output_dir = "graph_scores"
    unique_filename = get_unique_filename(output_dir, base_filename, extension)

    plt.savefig(unique_filename)



def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    
    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    labels (list, optional): List of labels to index the matrix. 
                             This may be used to reorder or select a subset of labels. 
                             If none is given, those that appear at least once in 
                             y_true or y_pred are used in sorted order.
    
    Returns:
    numpy.ndarray: Confusion matrix.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")
    
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    
    for yt, yp in zip(y_true, y_pred):
        cm[label_to_index[yt], label_to_index[yp]] += 1
    
    return cm