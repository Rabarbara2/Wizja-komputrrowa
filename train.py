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


def train_model(dataset_path: str):
    """
    Trenuje model KNN na danych z pliku CSV.

    Plik CSV powinien mieć format:
        label, cecha1, cecha2, cecha3, ...
    """

    df = pd.read_csv(dataset_path, header=None)

    df = df.dropna()

    # Jeśli label 0 oznacza błąd parsowania nazwy pliku, usuwamy takie wiersze.
    # W waszym przypadku klasy są 1-15, więc 0 nie powinno być poprawną klasą.
    df = df[df.iloc[:, 0] != 0]

    if df.empty:
        raise ValueError("Dataset jest pusty po usunięciu błędnych wierszy.")

    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    print("Liczba próbek:", len(df))
    print("Liczba cech:", X.shape[1])
    print("\nLiczność klas:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=69,
        stratify=y
    )

    scaler = StandardScaler()

    # Scaler uczymy tylko na zbiorze treningowym.
    X_train_scaled = scaler.fit_transform(X_train)

    # Zbiór testowy tylko transformujemy.
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(
        n_neighbors=3,
        metric="euclidean"
    )

    model.fit(X_train_scaled, y_train)

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


if __name__ == "__main__":
    dataset_path = input("Podaj ścieżkę do pliku CSV, np. dataset_teach.csv: ")

    if not dataset_path.strip():
        dataset_path = "dataset_teach.csv"

    train_model(dataset_path)