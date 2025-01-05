import copy
import pickle

import numpy as np
import os

import torch.optim

import torchmetrics
import logging
from snake import Direction
from torch.utils.data import Dataset, random_split
import torch.nn as nn
from torch.utils.data import DataLoader

"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""


def get_new_head(head: tuple, direction: Direction, block_size: int):
    if direction == Direction.UP:
        return head[0], head[1] - block_size
    elif direction == Direction.RIGHT:
        return head[0] + block_size, head[1]
    elif direction == Direction.DOWN:
        return head[0], head[1] + block_size
    elif direction == Direction.LEFT:
        return head[0] - block_size, head[1]


def is_collision(game_state: dict, block_size: int, bounds: tuple, action: Direction):
    head = game_state["snake_body"][-1]
    new_head = get_new_head(head, action, block_size)
    if new_head[0] < 0 or new_head[0] >= bounds[0] or new_head[1] < 0 or new_head[1] >= \
            bounds[1]:
        return True
    if new_head in game_state["snake_body"]:
        return True
    return False


def game_state_to_data_sample(game_state: dict, block_size: int, bounds: tuple):
    date_sample = np.zeros(8)
    for direction in Direction:
        if is_collision(game_state, block_size, bounds, direction):
            date_sample[direction.value] = 1

    if game_state["food"][1] < game_state["snake_body"][-1][1]:
        date_sample[4] = 1
    elif game_state["food"][1] > game_state["snake_body"][-1][1]:
        date_sample[6] = 1

    if game_state["food"][0] < game_state["snake_body"][-1][0]:
        date_sample[7] = 1
    elif game_state["food"][0] > game_state["snake_body"][-1][0]:
        date_sample[5] = 1

    return date_sample


def prepare_data(directory: str):
    """iterate through all pickle files in the directory, converts game state to date sample and merge them
     into a single data list"""
    data = []
    actions = []
    for file in os.listdir(directory):
        if file.endswith(".pickle"):
            with open(f"{directory + file}", 'rb') as f:
                data_file = pickle.load(f)
                block_size = data_file["block_size"]
                bounds = data_file["bounds"]
                for i in range(len(data_file["data"])):
                    game_state, action = data_file["data"][i]
                    if not is_collision(game_state, block_size, bounds, action):
                        data_sample = game_state_to_data_sample(game_state, block_size, bounds)
                        data.append(data_sample)
                        actions.append(action)
    return np.array(data), np.array(actions)


def convert_action_to_one_hot(action: Direction):
    one_hot = np.zeros(4)
    one_hot[action.value] = 1
    return one_hot


def convert_labels_to_one_hot(labels):
    new_labels = np.zeros((len(labels), 4))
    for idx in range(len(labels)):
        for direction in Direction:
            if labels[idx] == direction:
                new_labels[idx][direction.value] = 1.
    return new_labels


def convert_labels(labels):
    return np.array([direction.value for direction in labels])


class BCDataset(Dataset):
    def __init__(self, path):
        self.X, self.Y = prepare_data(path)
        self.Y = convert_labels_to_one_hot(self.Y)
        # self.Y = convert_labels(self.Y)
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MLP(nn.Module):
    def __init__(self, input_size, num_layer, layer_size, output_size, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.layers.append(nn.Linear(input_size, layer_size))
                self.layers.append(copy.deepcopy(activation))
            else:
                self.layers.append(nn.Linear(layer_size, layer_size))
                self.layers.append(copy.deepcopy(activation))

        self.layers.append(nn.Linear(layer_size, output_size))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_model(model, dataset, validation_set, batch_size, num_epochs, learning_rate):
    model.train()
    criterion = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="binary")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    logging.basicConfig(filename='training.log', level=logging.INFO)

    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.float()
            y = y.float()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy(output, y)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            for x, y in val_loader:
                x = x.float()
                y = y.float()
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                val_acc += accuracy(output, y)

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Save the model parameters if the current epoch gives a better validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_params = copy.deepcopy(model.state_dict())

        logging.info(
            f"Epoch: {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

    # Load the best model parameters
    model.load_state_dict(best_model_params)


if __name__ == "__main__":
    X, Y = prepare_data("data/")

    dataset = BCDataset("data/")

    train_size = int(0.8 * len(dataset))  # 80% danych treningowych
    val_size = int(0.1 * len(dataset))  # 10% danych walidacyjnych
    test_size = len(dataset) - train_size - val_size  # Reszta danych jako zbiór testowy

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    model = MLP(8, 3, 64, 4, nn.ReLU())
    train_model(model, train_dataset, val_dataset, 32, 100, 0.1)
    torch.save(model, "model.pth")
