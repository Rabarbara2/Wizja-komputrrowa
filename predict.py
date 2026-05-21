import os
import joblib

from leaf_analyzer import analyze_image
from zdjecia import analyze_image

MODEL_PATH = "models/knn_model.pkl"
SCALER_PATH = "models/scaler.pkl"


CLASS_NAMES = {
    2: "Acer - klon",
    4: "Quercus - dąb",
    7: "Salix alba - wierzba biała",
    8: "Populus tremula - topola osika",
    10: "Sorbus aucuparia - jarząb pospolity",
    12: "Populus - topola",
    13: "Tilia - lipa",
    15: "Fagus silvatica - buk",
}


def load_model_and_scaler():
    """
    Wczytuje wcześniej wytrenowany model KNN oraz scaler.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Nie znaleziono scalera: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def predict_leaf(image_path, show_debug=True):
    """
    Klasyfikuje pojedyncze zdjęcie liścia.

    Kroki:
    1. analiza obrazu,
    2. ekstrakcja cech,
    3. skalowanie cech,
    4. predykcja modelem KNN.
    """

    model, scaler = load_model_and_scaler()

    features = analyze_image(image_path)

    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]

    label = int(prediction)

    class_name = CLASS_NAMES.get(label, f"Nieznana klasa: {label}")

    return label, class_name


if __name__ == "__main__":
    image_path = input("Podaj ścieżkę do zdjęcia liścia: ")

    label, class_name = predict_leaf(
        image_path,
        show_debug=True
    )

    print("\nWynik klasyfikacji:")
    print(f"Label: {label}")
    print(f"Nazwa liścia: {class_name}")