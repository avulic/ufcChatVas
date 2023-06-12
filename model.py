import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader


num_folds = 5
num_epochs = 5
patience = 3
learning_rate = 0.1
batch_size = 2
model_path = "./models"
file_name = './data_edited.csv'

class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X=torch.tensor(X_train,dtype=torch.float32)
        self.y=torch.tensor(y_train,dtype=torch.float32)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len
    def columns(self):
        return self.X.shape[1]

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self):
        return len(self.indices)
    
class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        model_path = "./models"
        file_name = './data_edited.csv'
        # number of features (len of X cols)
        input_dim = input_size
        # number of hidden layers
        #hidden_layers = hidden_size
        # number of classes (unique of y)
        output_dim = output_size
        self.linear1 = nn.Linear(input_dim, 256)  # First fully connected layer
        self.linear2 = nn.Linear(256, 64)  # Second fully connected layer
        #self.linear3 = nn.Linear(128, 64)   # Third fully connected layer
        self.linear4 = nn.Linear(64, output_dim)     # Final output layer with 2 classes
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        #x = torch.sigmoid(self.linear3(x))
        x = self.linear4(x)
        return x

def LoadData(file_name):
    price_df=pd.read_csv(file_name)
    #columns = 884
    x=price_df.iloc[:,0:len(price_df.columns)-1].values
    y=price_df.iloc[:,len(price_df.columns)-1].values 
    X_train, X_test, Y_train, Y_test = train_test_split( x, y, test_size=0.30, random_state=42 )
    testData = Data(X_test, Y_test)
    trainData = Data(X_train, Y_train)
    
    return [testData, trainData]

def LoadModel(clf, model_path):
    clf.load_state_dict(torch.load(model_path))
    print("Neural Network Predictor Agent: Model loaded successfully")

class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_weights = model.state_dict()
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_model_weights = model.state_dict()
            self.counter = 0

    def get_best_model_weights(self):
        return self.best_model_weights

def train_model(model, train_data, plot=False):
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)
       
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # For fold results
    results = {}
    #for confusion matrix
    all_targets = []
    all_predictions = []
    all_probabilities = []

    try:
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data)):
            # Print
            #print(f'FOLD {fold}')
            #print('--------------------------------')
            
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            trainloader = DataLoader(
                            train_data, 
                            batch_size=batch_size, sampler=train_subsampler)
            testloader = DataLoader(
                            train_data,
                            batch_size=batch_size, sampler=test_subsampler)
            
            # Init the neural network
            network = model       
            # Initialize optimizer
            optimizer = optimizer
            
            # Run the training loop for defined number of epochs
            for epoch in range(0, num_epochs):
                # Print epoch
                #print(f'Starting epoch {epoch+1}')
                # Set current loss value
                current_loss = 0.0
                
                # Training phase
                model.train()
                # Iterate over the DataLoader for training data
                for i, data in enumerate(trainloader, 0):                
                    # Get inputs
                    inputs, targets = data 

                    # Zero the gradients
                    optimizer.zero_grad()                
                    # Perform forward pass
                    outputs = network(inputs)                
                    # Compute loss
                    loss = criterion(outputs, targets.long())
                    # Perform backward pass
                    loss.backward()                
                    # Perform optimization
                    optimizer.step()                
                    # Print statistics
                    current_loss += loss.item()
                    if i % 500 == 499:
                        #print('Loss after mini-batch %5d: %.3f' %(i + 1, current_loss / 500))
                        current_loss = 0.0
            # Process is complete.
            #print('Training process has finished. Saving trained model.')

            # Print about testing
            #print('Starting testing')
            # Saving the model
            save_path = f'./models/model-fold-{fold}.pth'
            torch.save(network.state_dict(), save_path)

            # Evaluationfor this fold
            model.eval()
            correct, total , val_loss = 0, 0, 0.0
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):
                    # Get inputs
                    inputs, targets = data
                    # Generate outputs
                    outputs = network(inputs)
                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    # Compute loss
                    loss = criterion(outputs, targets.long())
                    # Accumulate loss
                    val_loss += loss.item()

                    all_targets.extend(targets.numpy())
                    all_predictions.extend(predicted.numpy())

                    # Append probabilities for ROC curve
                    probabilities = nn.functional.softmax(outputs, dim=1)
                    all_probabilities.extend(probabilities[:, 1].numpy())

                accuracy = 100 * correct / total   
                # Calculate average validation loss
                val_loss /= len(testloader)
                # Call early_stopping
                early_stopping(val_loss, model)

                # Print accuracy
                #print('Accuracy for fold %d: %d %%' % (fold, accuracy))
                #print('--------------------------------')
                results[fold] = 100.0 * (correct / total)
                            
                if early_stopping.early_stop:
                    #print("Early stopping")
                    break
        
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum/len(results.items())} %')

        print("Neural Network Predictor Agent: Model trained successfully")
        if plot:
            best_model_weights = early_stopping.get_best_model_weights()
            model.load_state_dict(best_model_weights)

            plot_confusion_matrix(all_targets, all_predictions)
            # Plot ROC curve
            plot_roc_curve(all_targets, all_probabilities)
    except:
        print("Error in training")
        return False
    return True

def plot_confusion_matrix(all_targets, all_predictions):
    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    class_names = ['0', '1']
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(targets, probabilities):
    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()   

def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
    return predicted





