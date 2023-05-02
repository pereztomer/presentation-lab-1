import numpy as np
import torch
import torch.nn as nn
from lstm_model import LSTMClassifier
from data_utils import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import accuracy_score as sklearn_accuracy_score


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_seq_length = 336  # This is the maximun seq length from the train and test sets
    train_path = 'data/train'
    train_batch_size = 32
    train_shuffle = True
    train_ds = CustomDataset(folder_path=train_path, seq_length=50)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle)

    test_path = 'data/test'
    test_ds = CustomDataset(folder_path=test_path, seq_length=50)

    test_data_loader = DataLoader(dataset=test_ds,
                                  batch_size=train_batch_size,
                                  shuffle=train_shuffle)

    input_size = 40
    hidden_size = 20
    num_layers = 2
    output_size = 1
    num_epochs = 10
    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # input_data = torch.randn(32, 100, input_size)  # example input data
    # labels = torch.randint(0, 2, (32, 1)).float()  # example binary labels

    for epoch in range(num_epochs):
        # Train the model
        model.train()
        train_loss = 0.0
        train_predicted = []
        train_actual = []

        for batch_x, batch_labels in tqdm(train_data_loader):
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(torch.squeeze(outputs), batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy and F1 score
            predicted = torch.round(outputs)
            train_predicted += torch.squeeze(predicted).tolist()
            train_actual += batch_labels.tolist()

            # Add batch loss to total loss
            train_loss += loss.item() * batch_labels.size(0)

            # Calculate average training loss, accuracy and F1 score for the epoch
        train_loss /= len(train_ds)

        train_f1 = sklearn_f1_score(train_actual, train_predicted)
        train_accuracy = sklearn_accuracy_score(train_actual, train_predicted)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'train loss: {train_loss:.4f}, '
            f'train accuracy: {train_accuracy:.4f}, '
            f'train_f1: {train_f1:.4f}')


if __name__ == '__main__':
    train()
