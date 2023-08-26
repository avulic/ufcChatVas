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
from sklearn.preprocessing import MinMaxScaler

num_folds = 10
num_epochs = 9
patience = 3
learning_rate = 0.0001
batch_size = 30
model_path = "./models/models"
file_name = "./data/data_edited.csv"


class Data(Dataset):
    def __init__(self, X_train, y_train):
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # X_train = scaler.fit_transform(X_train)

        self.X = torch.tensor(X_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.float32)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

    def columns(self):
        return self.X.shape[1]


class Network(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.5):
        super(Network, self).__init__()

        input_dim = input_size
        output_dim = output_size

        # Define linear layers
        self.linear1 = nn.Linear(input_dim, 500)
        self.linear2 = nn.Linear(500, 400)
        self.linear3 = nn.Linear(400, output_dim)

        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(500)
        self.bn2 = nn.BatchNorm1d(400)

        # Define dropout layers
        self.dropout = nn.Dropout(p=dropout_prob)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)  # Apply dropout after the activation

        x = self.linear2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)  # Apply dropout after the activation

        x = torch.sigmoid(self.linear3(x))
        return x


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


def LoadData(file_name):
    price_df = pd.read_csv(file_name)
    x = price_df.drop("Winner", axis=1).values
    y = price_df["Winner"].values
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )
    testData = Data(X_test, Y_test)
    trainData = Data(X_train, Y_train)

    return [testData, trainData]


def LoadModel(clf, model_path):
    clf.load_state_dict(torch.load(model_path))
    print("Neural Network Predictor Agent: Model loaded successfully")


def LoadModel(clf, early_stopping):
    best_model_weights = early_stopping.get_best_model_weights()
    clf.load_state_dict(best_model_weights)
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
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # For fold results
    results = {}

    training_losses = []
    validation_losses = []

    try:
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data)):
            # Print
            print(f"FOLD {fold}")
            # print('--------------------------------')

            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)

            trainloader = DataLoader(
                train_data, batch_size=batch_size, sampler=train_subsampler
            )
            testloader = DataLoader(
                train_data, batch_size=batch_size, sampler=test_subsampler
            )

            input_size = train_data.columns()
            output_size = 1
            network = Network(input_size, output_size)

            criterion = nn.BCELoss()
            optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
            early_stopping = EarlyStopping(patience=patience)

            # Run the training loop for defined number of epochs
            # print('Starting training')
            for epoch in range(0, num_epochs):
                current_loss = 0.0
                # Training phase
                network.train()

                print("Starting epoch:" + str(epoch))
                for i, data in enumerate(trainloader, 0):
                    inputs, targets = data

                    optimizer.zero_grad()
                    outputs = network(inputs)

                    loss = criterion(outputs.view(-1), targets)
                    loss.backward()
                    optimizer.step()
                    current_loss += loss.item()
                    if i % 500 == 499:
                        print(
                            "Loss after mini-batch %5d: %.3f"
                            % (i + 1, current_loss / 500)
                        )
                        current_loss = 0.0

            # Append training loss for this epoch
            training_losses.append(current_loss / len(trainloader))

            # print('Starting testing')
            save_path = f"./models/model-fold-{fold}.pth"
            torch.save(network.state_dict(), save_path)

            # Evaluationfor this fold
            network.eval()
            correct, total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):
                    inputs, targets = data
                    outputs = network(inputs)

                    predicted = (outputs.data > 0.5).float()

                    total += targets.size(0)
                    correct += (predicted.squeeze() == targets).sum().item()

                    loss = criterion(outputs.view(-1), targets)
                    val_loss += loss.item()

                accuracy = 100 * correct / total
                val_loss /= len(testloader)

                validation_losses.append(val_loss)

                early_stopping(val_loss, network)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # Print accuracy
            print("Accuracy for fold %d: %d %%" % (fold, accuracy))
            print("--------------------------------")
            results[fold] = 100.0 * (correct / total)

        print(f"K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS")
        print("--------------------------------")
        sum = 0.0
        for key, value in results.items():
            print(f"Fold {key}: {value} %")
            sum += value
        print(f"Average: {sum/len(results.items())} %")

        print("Neural Network Predictor Agent: Model trained successfully")
        LoadModel(network, early_stopping)

        plot_Error(training_losses, validation_losses)

    except Exception as e:
        print("Error in training: " + str(e))
        return False
    return True


def plot_confusion_matrix(all_targets, all_predictions):
    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    class_names = ["0", "1"]
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(targets, probabilities):
    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_Error(training_losses, validation_losses=None):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss")
    if validation_losses != None:
        plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
    return predicted


def test_model(model, test_data):
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model.eval()  # Set the model to evaluation mode
    test_correct = 0
    test_total = 0
    all_test_targets = []
    all_test_predictions = []
    all_test_probabilities = []
    test_losses = []

    criterion = nn.BCELoss()

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            outputs = model(inputs)

            predicted = (outputs.data > 0.5).float()

            test_total += targets.size(0)
            test_correct += (predicted.squeeze() == targets).sum().item()

            all_test_targets.extend(targets.numpy())

            # Append probabilities for ROC curve
            probabilities = outputs

            all_test_predictions.extend(predicted.numpy())
            all_test_probabilities.extend(probabilities.numpy())

            loss = criterion(outputs.view(-1), targets)
            test_losses.append(loss.item())

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Confusion Matrix
    plot_confusion_matrix(all_test_targets, all_test_predictions)

    # Plot ROC curve
    plot_roc_curve(all_test_targets, all_test_probabilities)

    # Plot Test Loss
    # plot_Error(test_losses)


if __name__ == "__main__":
    [trainData, testData] = LoadData(file_name)
    trainData = trainData
    input_size = trainData.columns()
    output_size = 1
    mreza = Network(input_size, output_size)
    fighters = pd.read_csv("data/fighters.csv")
    fighter_mean = pd.read_csv("data/fighter_means.csv")

    train_model(mreza, trainData)
    test_model(mreza, testData)
