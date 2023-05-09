import numpy as np
import torch
import torch.nn as nn
from lstm_model import LSTMClassifier
from data_utils import CustomDataset, plot_graph
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import accuracy_score as sklearn_accuracy_score, precision_score, recall_score
from transformer_model import TransformerClassifier


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    max_seq_length = 336  # This is the maximum seq length from the train and test sets
    train_path = 'original_data/train'
    train_batch_size = 32
    train_shuffle = True
    train_ds = CustomDataset(folder_path=train_path, seq_length=100)

    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_size=train_batch_size,
                                   shuffle=train_shuffle)

    test_path = 'original_data/test'
    test_ds = CustomDataset(folder_path=test_path, seq_length=100)

    test_data_loader = DataLoader(dataset=test_ds,
                                  batch_size=train_batch_size,
                                  shuffle=False)
    num_epochs = 50

    # lstm params:
    # input_size = 26
    # hidden_size = 100
    # num_layers = 2
    # output_size = 1
    # model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)

    # transformer params:
    input_size = 26
    hidden_size = 40
    num_layers = 2
    output_size = 1
    num_heads = 4
    dropout = 0.3

    model = TransformerClassifier(input_size,
                                  hidden_size,
                                  num_layers,
                                  output_size,
                                  num_heads,
                                  dropout).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # input_data = torch.randn(32, 100, input_size)  # example input data
    # labels = torch.randint(0, 2, (32, 1)).float()  # example binary labels

    train_loss_list = []
    test_loss_list = []

    train_f1_list = []
    test_f1_list = []

    train_accuracy_list = []
    test_accuracy_list = []

    train_recall_list = []
    test_recall_list = []

    train_precision_list = []
    test_precision_list = []

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
        train_precision = precision_score(train_actual, train_predicted)
        train_recall = recall_score(train_actual, train_predicted)
        # Evaluate the model on the test set
        model.eval()
        test_total_loss = 0.0
        test_total_predicted = []
        test_total_actual = []

        with torch.no_grad():
            for test_batch_x, test_batch_labels in tqdm(test_data_loader):
                test_batch_x = test_batch_x.to(device)
                test_batch_labels = test_batch_labels.to(device)

                test_outputs = model(test_batch_x)
                test_loss = criterion(torch.squeeze(test_outputs), test_batch_labels)
                # Calculate training accuracy and F1 score
                test_predicted = torch.round(test_outputs)
                test_total_predicted += torch.squeeze(test_predicted).tolist()
                test_total_actual += test_batch_labels.tolist()

                # Add batch loss to total loss
                test_total_loss += test_loss.item() * test_batch_labels.size(0)

        # Calculate average training loss, accuracy and F1 score for the epoch
        test_total_loss /= len(test_ds)

        test_f1 = sklearn_f1_score(test_total_actual, test_total_predicted)
        test_accuracy = sklearn_accuracy_score(test_total_actual, test_total_predicted)
        test_precision = precision_score(test_total_actual, test_total_predicted)
        test_recall = recall_score(test_total_actual, test_total_predicted)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'train loss: {train_loss:.4f}, '
            f'train accuracy: {train_accuracy:.4f}, '
            f'train f1: {train_f1:.4f}\n'
            f'test loss:{test_loss:.4f} '
            f'test accuracy: {test_accuracy:.4f} '
            f'test f1: {test_f1:.4f} '
            f'test recall: {test_recall:.4f} '
            f'test precision: {test_precision:.4f}')

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss.cpu().item())

        train_f1_list.append(train_f1)
        test_f1_list.append(test_f1)

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

        train_recall_list.append(train_recall)
        test_recall_list.append(test_recall)

        train_precision_list.append(train_precision)
        test_precision_list.append(test_precision)

    plot_graph(train_loss_list,
               test_loss_list,
               'train loss',
               'test loss',
               'train/test loss vs epoch')
    plot_graph(train_f1_list,
               test_f1_list,
               'train f1',
               'test f1',
               'train/test f1 vs epoch')
    plot_graph(train_accuracy_list,
               test_accuracy_list,
               'train accuracy',
               'test accuracy',
               'train/test accuracy vs epoch')
    plot_graph(train_precision_list,
               test_precision_list,
               'train precision',
               'test precision',
               'train/test precision vs epoch')
    plot_graph(train_recall_list,
               test_recall_list,
               'train recall',
               'test recall',
               'train/test recall vs epoch')


if __name__ == '__main__':
    train()
