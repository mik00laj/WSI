import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from NaiveBayes import NaiveBayes


def load_and_prepare_data(train_data_path, test_data_path):
    data_train = pd.read_csv(train_data_path)
    data_test = pd.read_csv(test_data_path)

    # Przygotowanie danych treningowych
    data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())
    data_train['Embarked'] = data_train['Embarked'].fillna('S')
    data_train['Fare'] = data_train['Fare'].fillna(data_train['Fare'].median())

    # Przygotowanie danych testowych
    data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
    data_test['Embarked'] = data_test['Embarked'].fillna('S')
    data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].median())

    # Tworzenie grup wiekowych
    data_train['AgeGroup'] = pd.cut(data_train['Age'], bins=[0, 12, 18, 65, 100],
                                    labels=['Child', 'Teen', 'Adult', 'Senior'])
    data_test['AgeGroup'] = pd.cut(data_test['Age'], bins=[0, 12, 18, 65, 100],
                                   labels=['Child', 'Teen', 'Adult', 'Senior'])

    return data_train, data_test


def preprocess_data(data_train, data_test, features, target):
    # Konwersja zmiennych kategorycznych na numeryczne
    data_train = pd.get_dummies(data_train[features + [target]], drop_first=True)
    data_test = pd.get_dummies(data_test[features], drop_first=True)

    # Sprawdzenie czy  kolumny w danych testowych są takie same jak w danych treningowych
    missing_cols = set(data_train.columns) - set(data_test.columns) - {target}
    for c in missing_cols:
        data_test[c] = 0
    data_test = data_test[data_train.columns.drop(target)]

    return data_train, data_test


def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    nb = NaiveBayes()
    # nb = GaussianNB()

    nb.fit(X_train, y_train)

    y_pred_val = nb.predict(X_val)
    y_pred_train = nb.predict(X_train)

    accuracy_val = np.sum(y_pred_val == y_val) / len(y_val)
    accuracy_train = np.sum(y_pred_train == y_train) / len(y_train)

    print(f"Dokładność na zbiorze walidacyjnym: {accuracy_val}")
    print(f"Dokładność na zbiorze treningowym: {accuracy_train}")

    report_val = classification_report(y_val, y_pred_val)
    report_train = classification_report(y_train, y_pred_train)

    print("Classification Report dla zbioru walidacyjnego:")
    print(report_val)
    print("Classification Report dla zbioru treningowego:")
    print(report_train)

    return nb, y_pred_train, y_pred_val


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dead', 'survived'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

#Funkcja ta przekształca cechy pasażera na wektor cech zgodny z danymi treningowymi,
#a następnie używa modelu Naive Bayes do obliczenia prawdopodobieństwa przeżycia.

def sa(nb, conditions, data_train, target):
    sample = pd.DataFrame([conditions])
    sample = pd.get_dummies(sample, drop_first=True).reindex(columns=data_train.drop(target, axis=1).columns,
                                                             fill_value=0)
    return nb.predict_proba(sample.values)[0][1]


def main():
    train_data_path = "train.csv"
    test_data_path = "test.csv"
    features = ['Pclass', 'Sex', 'AgeGroup', 'Fare', 'Embarked']
    target = 'Survived'

    # Przygotowanie danych
    data_train, data_test = load_and_prepare_data(train_data_path, test_data_path)
    data_train, data_test = preprocess_data(data_train, data_test, features, target)

    # Podział danych treningowych na treningowe i walidacyjne
    X = data_train.drop(target, axis=1).values
    y = data_train[target].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trenowanie własnego klasyfikatora Naive Bayes
    nb, y_pred_train, y_pred_val = train_and_evaluate_model(X_train, y_train, X_val, y_val)

    # Rysowanie macierzy konfuzji
    plot_confusion_matrix(y_train, y_pred_train, "Confusion Matrix - Training Data")
    plot_confusion_matrix(y_val, y_pred_val, "Confusion Matrix - Validation Data")

    # Obliczanie prawdopodobieństw dla różnych warunków
    conditions = {
        "Pclass": 1,
        "Sex_male": 1,
        "AgeGroup_Teen": 0,
        "AgeGroup_Adult": 1,
        "AgeGroup_Senior": 0,
        "Fare": 50,
        "Embarked_Q": 0,
        "Embarked_S": 1
    }

    print("Prawdopodobieństwo przeżycia dla mężczyzny:", predict_survival_proba(nb, conditions, data_train, target))

    conditions["Sex_male"] = 0
    print("Prawdopodobieństwo przeżycia dla kobiety:", predict_survival_proba(nb, conditions, data_train, target))

    conditions.update({"AgeGroup_Child": 1, "AgeGroup_Adult": 0})
    print("Prawdopodobieństwo przeżycia dla chłopca:", predict_survival_proba(nb, conditions, data_train, target))

    conditions["Sex_male"] = 0
    print("Prawdopodobieństwo przeżycia dla dziewczynki:", predict_survival_proba(nb, conditions, data_train, target))

    # Tworzenie pliku gender_submission.csv
    y_pred_test = nb.predict(data_test.values)
    submission = pd.DataFrame({'PassengerId': pd.read_csv(test_data_path)['PassengerId'], 'Survived': y_pred_test})
    submission.to_csv('gender_submission.csv', index=False)

    print("Plik gender_submission.csv został wygenerowany.")


if __name__ == "__main__":
    main()
