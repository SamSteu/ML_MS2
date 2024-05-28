import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.utils import accuracy_fn, onehot_to_label, macrof1_fn
from tqdm import tqdm

## MS2

 
class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    
    def __init__(self, input_size, n_classes, dropout_prob=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes))


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


class CNN(nn.Module):
    """
    A CNN which does classification on the Fashion MNIST dataset.
    """

    def __init__(self, input_channels, n_classes, filters=(32, 64, 128), dropout_rate=0.5):
        """
        Initialize the network.

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
            filters (tuple): number of filters for each convolutional layer
            dropout_rate (float): dropout rate to use in Dropout layers
        """
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=filters[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        self.fc1 = nn.Linear(filters[2] * 3 * 3, 256)  # Adjust according to the downsampled feature map size
        self.fc2 = nn.Linear(256, n_classes)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        
        return x







class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):       #A VOIR LES PARAMS PAR DEFAUT (à optimiser)
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq) ### WRITE YOUR CODE HERE

                attention = self.softmax(q @ k.T / np.sqrt(self.d_head))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            #nn.Linear(mlp_ratio * hidden_d, hidden_d),
            #nn.GELU()
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        #out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super(MyViT, self).__init__()
        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches**2+1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )


    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w # We assume square image.

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):

                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]

                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()

        return patches


    def get_positional_embeddings(self, sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = torch.sin(torch.tensor(i / (10000 ** (j / d)), dtype=torch.float32))
                else:
                    result[i, j] = torch.cos(torch.tensor(i / (10000 ** ((j-1) / d)), dtype=torch.float32))
        return result


    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        n, c, h, w = x.shape

        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out
        #return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, optimizer_name):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()

        if optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        elif optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")


        self.accuracy_list = []
        self.macrof1_list = []


    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """

        for ep in range(self.epochs):
            print("train epoch : ", ep, " / ", self.epochs)
            self.train_one_epoch(dataloader, ep)
            
            correct = 0
            total = 0
            self.model.eval()
            
            # Disable gradient calculation
            with torch.no_grad():
                for batch in dataloader:
                    inputs, labels = batch
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Calculate accuracy and F1 score
            acc = correct / total * 100
            macrof1 = macrof1_fn(predicted, labels)

            self.accuracy_list.append(acc)
            self.macrof1_list.append(macrof1*100)

            #idéalement il faudrait print un truc dans ce style : "Ep 1/20, it 469/469: loss train: 1.91, accuracy train: 0.55, accuracy test: 0.53" (cf série 11 tranformers)
            print(f"\nTrain set: accuracy = {acc:.3f}%, F1-score = {macrof1:.6f}")

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        

        self.model.train()
        for it, batch in tqdm(enumerate(dataloader)):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch

            # 5.2 Run forward pass.
            logits = self.model(x) 
            # 5.3 Compute loss (using 'criterion').
            loss = self.criterion(logits, y)
            # 5.4 Run backward pass.
            loss.backward()
            # 5.5 Update the weights using 'optimizer'.
            self.optimizer.step() 
            
            # 5.6 Zero-out the accumulated gradients.
            self.optimizer.zero_grad() 
        return dataloader                       




    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """

        self.model.eval()  # Set model to evaluation mode
        pred_labels = []

        with torch.no_grad():  # Disable gradient calculation
            for batch in dataloader:
                x = batch[0]  # Get the input data from the batch
                outputs = self.model(x)  # Get the model predictions
                _, preds = torch.max(outputs, 1)  # Get the predicted class labels
                pred_labels.append(preds)

        # Concatenate all the predicted labels into a single tensor
        pred_labels = torch.cat(pred_labels)

        return pred_labels
    



    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        #fitting
        self.train_all(train_dataloader)

        return self.predict(training_data)





    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
    
    def get_logits(self, data) :
        self.model.eval()
        dataset = TensorDataset(torch.from_numpy(data).float())
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_logits = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                outputs = self.model(x)
                all_logits.append(outputs)

        all_logits = torch.cat(all_logits)
        return all_logits.cpu().numpy()