import numpy as np 
import matplotlib.pyplot as plt

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



def ROC_curve(probas, true, name) :
    plt.figure(figsize=(9,4))
    plt.title(f"ROC curve of {name}")
    plt.ylabel("True Positive rate")
    plt.xlabel("False Positive rate")
    tresholdsSet = np.arange(0, 1, 0.1)
    areas = 0

    for item in range(probas.shape[1]) :
        X_candidates = []
        Y_candidates = []
        for treshold in tresholdsSet :
            P = np.sum(true[:, item])
            N = true.shape[0] - P

            oneColumnPredicted = probas[:, item]
            oneColumnPredicted = np.where(oneColumnPredicted >= treshold, 1, 0)

            oneColumnTrue = true[:, item]

            TP = np.sum((oneColumnPredicted == 1) & (oneColumnTrue == 1))
            FP = np.sum((oneColumnPredicted == 1) & (oneColumnTrue == 0))
            FN = np.sum((oneColumnPredicted == 0) & (oneColumnTrue == 1))
            TN = np.sum((oneColumnPredicted == 0) & (oneColumnTrue == 0))

            TPR = TP / float(P)
            FPR = FP / float(N)


            X_candidates.append(FPR)
            Y_candidates.append(TPR)

        areas -= np.trapz(Y_candidates, X_candidates)
        plt.plot(X_candidates, Y_candidates)


    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), '--', color = "gray", label = 'y = x')
    areas /= float(probas.shape[1])
    plt.text(0.5, 0.0, f'AUC = {areas}', ha='center')
    plt.legend()
    plt.show()


def plot_epoch_score(epoch_acc, epoch_f1, titre, acc_train, macrof1_train, acc_test, macrof1_test):
    print("Scores during training phase")
    n = len(epoch_acc)
    plt.figure(figsize=(9,4))
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
    plt.show()
