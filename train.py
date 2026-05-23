import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Nazwy plików, w których zostanie zapisany wytrenowany model i scaler.
# Scaler przydaje się, aby skalować nowe dane w ten sam sposób co dane treningowe.

def train_model(dataset_path: str):
    """
    Trenuje model KNN na danych z pliku CSV.

    Plik CSV powinien mieć format:
        label, cecha1, cecha2, cecha3, ...
    """

    # Wczytujemy dane z pliku CSV bez nagłówka, zakładając że pierwsza kolumna to etykiety klas.
    df = pd.read_csv(dataset_path, header=None)

    # Usuwamy wszystkie wiersze zawierające wartości brakujące, bo nie można ich użyć do uczenia.
    df = df.dropna()

    # Jeśli label 0 oznacza błąd parsowania nazwy pliku, usuwamy takie wiersze.
    # W tym projekcie prawidłowe klasy zaczynają się od 1, więc 0 jest odrzucane.
    df = df[df.iloc[:, 0] != 0]

    if df.empty:
        raise ValueError("Dataset jest pusty po usunięciu błędnych wierszy.")

    # Oddzielamy etykiety (y) od cech (X). Pierwsza kolumna to etykieta klasy.
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # Używamy wartości bezwzględnej, jeśli cechy mogą zawierać ujemne wartości.
    X = X.abs()

    print("Liczba próbek:", len(df))
    print("Liczba cech:", X.shape[1])
    print("\nLiczność klas:")
    print(y.value_counts().sort_index())

    # Dzielimy dane na zbiór treningowy i testowy. Stratify=y zachowuje rozkład klas.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=69,
        stratify=y
    )

    scaler = StandardScaler()

    # Uczymy scaler tylko na danych treningowych, aby uniknąć wycieku informacji z testu.
    X_train_scaled = scaler.fit_transform(X_train)

    # Zastosowanie tego samego przeskalowania do danych testowych.
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(
        n_neighbors=3,
        metric="euclidean"
    )

    # Trenujemy klasyfikator KNN na przeskalowanych cechach treningowych.
    model.fit(X_train_scaled, y_train)

    # Przewidujemy etykiety dla danych testowych.
    y_pred = model.predict(X_test_scaled)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nModel zapisany do: {MODEL_PATH}")
    print(f"Scaler zapisany do: {SCALER_PATH}")
    return accuracy_score(y_test, y_pred)