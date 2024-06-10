import os
import pickle
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from snake import Direction

class NaiveBayes(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Mean, variance, and prior probabilities
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-6  # Add small value to avoid division by zero
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        if not np.issubdtype(x.dtype, np.floating):
            x = np.array(x, dtype=np.float64)
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

def game_state_to_data_sample(game_state):
    snake_body = game_state['snake_body']
    food_position = game_state['food']
    snake_direction = game_state['snake_direction']

    # Inicjalizacja wektora cech
    features = []
    head_x, head_y = snake_body[-1]
    default_bounds = (300, 300)
    bounds = game_state.get('bounds', default_bounds)

    # Cechy dotyczące przeszkód
    obstacles = {
        'left': any((head_x - 30, head_y) == segment for segment in snake_body) or head_x <= 0,
        'right': any((head_x + 30, head_y) == segment for segment in snake_body) or head_x >= bounds[0] - 30,
        'up': any((head_x, head_y - 30) == segment for segment in snake_body) or head_y <= 0,
        'down': any((head_x, head_y + 30) == segment for segment in snake_body) or head_y >= bounds[1] - 30,
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

    # Dodanie aktualnego kierunku ruchu węża jako cechy
    direction_feature = [0, 0, 0, 0]
    if snake_direction == Direction.UP:
        direction_feature[0] = 1
    elif snake_direction == Direction.RIGHT:
        direction_feature[1] = 1
    elif snake_direction == Direction.DOWN:
        direction_feature[2] = 1
    elif snake_direction == Direction.LEFT:
        direction_feature[3] = 1

    features.extend(direction_feature)

    return np.array(features)

def load_data():
    files = os.listdir('data')
    data = []
    for file in files:
        with open(os.path.join('data', file), 'rb') as f:
            data.extend(pickle.load(f)['data'])

    X = []
    y = []
    for game_state, direction in data:
        X.append(game_state_to_data_sample(game_state))
        y.append(direction.value)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_data()
    model = NaiveBayes()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracy_train = np.sum(y_pred_train == y_train) / len(y_train)
    accuracy_test = np.sum(y_pred_test == y_test) / len(y_test)

    print(f"Dokładność na zbiorze treningowym: {accuracy_train}")
    print(f"Dokładność na zbiorze testowym: {accuracy_test}")

    report_train = classification_report(y_train, y_pred_train)
    report_test = classification_report(y_test, y_pred_test)

    print("Classification Report dla zbioru treningowego:")
    print(report_train)
    print("Classification Report dla zbioru testowego:")
    print(report_test)
