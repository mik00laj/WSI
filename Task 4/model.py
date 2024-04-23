import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def load_data(data_directory='data'):
    all_data = []
    labels = []

    # Przeglądanie folderu z danymi i odczyt wszystkich plików .pickle
    for filename in os.listdir(data_directory):
        if filename.endswith('.pickle'):
            filepath = os.path.join(data_directory, filename)
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
                for game_state, action in data['data']:
                    print(game_state)  # Sprawdzanie, czy zawiera 'bounds'
                    features = game_state_to_data_sample(game_state)
                    all_data.append(features)
                    labels.append(action.value)  # Zakładając, że `action` to instancja Enum

    return np.array(all_data), np.array(labels)


def game_state_to_data_sample(game_state: dict):
    snake_body = game_state['snake_body']
    food_position = game_state['food']
    snake_direction = game_state['snake_direction']

    # Inicjalizacja wektora cech
    features = []
    head_x, head_y = snake_body[-1]
    default_bounds = (300, 300)  #
    bounds = game_state.get('bounds', default_bounds)

    # Cechy dotyczące przeszkód
    obstacles = {
        'left': any((head_x - 1, head_y) == segment for segment in snake_body) or head_x == 0,
        'right': any((head_x + 1, head_y) == segment for segment in snake_body) or head_x == bounds[0] - 1,
        'up': any((head_x, head_y - 1) == segment for segment in snake_body) or head_y == 0,
        'down': any((head_x, head_y + 1) == segment for segment in snake_body) or head_y == bounds[1] - 1,
    }
    features.extend(obstacles.values())

    # Cechy dotyczące lokalizacji jedzenia
    food_direction = {
        'food_left': food_position[0] < head_x,
        'food_right': food_position[0] > head_x,
        'food_up': food_position[1] < head_y,
        'food_down': food_position[1] > head_y,
    }
    features.extend(food_direction.values())

    return features

class SVM:
    def __init__(self, C=1.0):
        self.weights = None
        self.bias = None
        self.C = C

    def fit(self, X, y, epochs=1000, learning_rate=0.001):
        # Pobieramy liczbę próbek i cech z macierzy
        n_samples, n_features = X.shape
        #inicjalizujemy wagi i obciążenie na początkowe wartości zerowe
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y <= 0, -1, 1)
        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= learning_rate * (2 * self.weights)
                else:
                    # aktualizowane na podstawie gradientu funkcji straty
                    self.weights -= learning_rate * (2 * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= learning_rate * y_[idx]
            # Aktualizacja wag w zależności od parametru C
            self.weights -= 1 / self.C * self.weights

    # Przewidywanie klas na nowych danych
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)


def train_and_evaluate_model(percentage, C):

    X, y = load_data()

    # Wybór podzbioru danych
    subset_size = int(len(X) * percentage)
    X_subset = X[:subset_size]
    y_subset = y[:subset_size]

    # Podział podzbioru danych na zbiór treningowy i testowy w prporpocj 4:1
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

    # Inicjalizacja i trenowanie modelu
    # model = SVM()
    # model = SVM(C=C)
    # model = SVC()
    model = SVC(C=C)
    model.fit(X_train, y_train)

    # Testowanie modelu za zbiorze testowym
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Zbiór danych testowych")
    print(f"Rozmiar zbioru danych: {percentage * 100}%")
    print(f"Parametr C: {C}")
    print("Dokładność modelu:", accuracy)

    # Testowanie modelu za zbiorze treningowym
    predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, predictions)
    print("Zbiór danych trenigowych")
    print(f"Rozmiar zbioru danych: {percentage * 100}%")
    print(f"Parametr C: {C}")
    print("Dokładność modelu:", accuracy)


if __name__ == "__main__":
    # percentage = 0.01
    # percentage = 0.1
    percentage = 1.0
    # C = 0.01
    # C = 0.1
    C = 1
    # C = 10
    # C =100
    train_and_evaluate_model(percentage, C)
