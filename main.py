import argparse
import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, ROC_curve, visualize_histogram, plot_epoch_score

import copy
import torch
import matplotlib.pyplot as plt
np.random.seed(100)
import time

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """

    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)

    print("xtrain : ", xtrain.shape)
    print("ytrain : ", ytrain.shape)
    print("xtest : ", xtest.shape)
    
    

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    #normalize : 
    means = np.mean(xtrain)
    std = np.std(xtrain)
    xtrain = normalize_fn(xtrain, means, std)

    means = np.mean(xtest)
    std = np.std(xtest)
    xtest = normalize_fn(xtest, means, std)

    #add bias :
    # à demander aux assistants pour quels deep network on met un biais (car CNN on aura plus du 28x28 ca pose pb)

    #global variables :
    n_samples = xtrain.shape[0]
    n_features = xtrain.shape[1]

    # Make a validation set
    if not args.test:
    ### WRITE YOUR CODE HERE
        all_ind = np.arange(n_samples)
        rdm_perm_ind = np.random.permutation(all_ind)
        n_test = int(n_samples * args.val_set)

        xtrain_temp = copy.deepcopy(xtrain)
        xtrain = xtrain_temp[rdm_perm_ind[:n_test]]
        xtest = xtrain_temp[rdm_perm_ind[n_test:]]

        ytrain_temp = copy.deepcopy(ytrain)
        ytrain = ytrain_temp[rdm_perm_ind[:n_test]]
        ytest = ytrain_temp[rdm_perm_ind[n_test:]]

        #A DECOMMENTER POUR LE RENDU !!
        #visualize_histogram(ytrain, ytest)
    
    
    
    
    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca and args.nn_type == "mlp":
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        exvar = pca_obj.find_principal_components(xtrain)
        exvar2 = pca_obj.find_principal_components(xtest)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        print(xtrain.shape)
        print(f'The total variance explained by the first {args.pca_d} principal components is {exvar} %')

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(xtrain.shape[1], n_classes, dropout_prob=args.dropout)
    
    elif args.nn_type == "cnn" :
        model = CNN(1, n_classes)  #(input_channels, n_classes, filters=(16, 32, 64))
        
        #j'ai reshape en 3 dimensions mais je suis pas sure
        xtrain = xtrain.astype(np.float32).reshape(xtrain.shape[0], 1, 28, -1)
        xtest = xtest.astype(np.float32).reshape(xtest.shape[0], 1, 28, -1)

    
    elif args.nn_type == "transformer" :
        xtrain = xtrain.astype(np.float32).reshape(xtrain.shape[0], 1, 28, -1)
        xtest = xtest.astype(np.float32).reshape(xtest.shape[0], 1, 28, -1)
        #VALEURS DES PARAMS A AJUSTER
        model = MyViT((1, 28, 28), n_patches = 7, n_blocks = 1, hidden_d = 256, n_heads = 8, out_d =10)

    summary(model)

    # Trainer object
    print("starting instantiate Trainer...")
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    print("instantiated Trainer !\n")


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    print("fitting Trainer...")
    preds_train = method_obj.fit(xtrain, ytrain)
    print("fitted Trainer !\n")



    # Predict on unseen data
    print("starting to predict Trainer on unseen data...")
    preds = method_obj.predict(xtest)
    print("finished predict Trainer !\n")


    ## Report results: performance on train and valid/test sets
    acc_train = accuracy_fn(preds_train, ytrain)
    print(ytrain.shape)
    macrof1_train = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}")

    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc_test = accuracy_fn(preds, ytest)         #j'ai remplacé, avant c'était xtest
    macrof1_test = macrof1_fn(preds, ytest)      #idem
    print(f"Validation set:  accuracy = {acc_test:.3f}% - F1-score = {macrof1_test:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.

    # Ce serait cool d'ajouter ROC curve, Confusion matrix commme pour le MS1

    #plot les accuracies pour voir l'évolution au cours des epochs
    plot_epoch_score(method_obj.accuracy_list, method_obj.macrof1_list, args.title, acc_train, macrof1_train, acc_test, macrof1_test)




if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=84, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    
    #our arguments :
    parser.add_argument("--val_set", type = float, default = 0.8, help = "percentage of validation set")
    parser.add_argument("--title", type = str, default = "sans titre", help = "titre pour plot train accuracy et F1")
    parser.add_arguments("--dropout", type = float, default = 0.3, help = "dropout percentage for MLP")
    parser.add_argument("--filter", type=float, nargs=3, help="filter parameters for CNN ")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    s1 = time.time()
    main(args)
    s2 = time.time()
    print(f"Running time : {s2-s1} seconds")