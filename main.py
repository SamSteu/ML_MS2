import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
import copy
import torch


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """

    #peut être utile maybe ?
    class_names = ['0 Top/T-Shirt', '1 Trouser', '2 Pullover', '3 Drees', '4 Coat', '5 Sandal', '6 Shirt', '7 Sneaker', '8 Bag', '9 Ankle Boot']


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

    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(xtrain.shape[1], n_classes)
        args.nn_batch_size = 1
    
    elif args.nn_type == "cnn" :
        model = CNN(1, n_classes)  #(input_channels, n_classes, filters=(16, 32, 64))
        
        #j'ai reshape en 3 dimensions mais je suis pas sure
        xtrain = xtrain.astype(np.float32).reshape(xtrain.shape[0], 1, 28, -1)
        xtest = xtest.astype(np.float32).reshape(xtest.shape[0], 1, 28, -1)

        
    
    elif args.nn_type == "transformer" :
        model = MyViT(chw, n_patches, n_blocks, hidden_d, n_heads, out_d)

    summary(model)

    # Trainer object
    print("start instantiate Trainer")
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    print("instantiated Trainer")


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    print("fit Trainer")
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    print("start to predict Trainer")
    preds = method_obj.predict(xtest)
    print("finish predict Trainer")

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, xtest)
    macrof1 = macrof1_fn(preds, xtest)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


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
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    
    #our arguments :
    parser.add_argument("--val_set", type = float, default = 0.8, help = "percentage of validation set")


    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)