import os
import joblib

from zdjecia import analyze_image

MODEL_PATH = "models/knn_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Słownik z mapowaniem numerów klas na ich czytelną nazwę.
# Używany do wyświetlenia nazwy liścia zamiast samej etykiety liczbowej.
CLASS_NAMES = {
    1: "Ulmus carpinifolia - wiąz pospolity",
    2: "Acer - klon",
    4: "Quercus - dąb",
    6: "Betula pubescens - brzoza omszona",
    7: "Salix alba - wierzba biała",
    8: "Populus tremula - topola osika",
    9: "Ulmus grabla - wiąz górski",
    10: "Sorbus aucuparia - jarząb pospolity",
    11: "Salix cinerea - wierzba szara",
    12: "Populus - topola",
    13: "Tilia - lipa"
}

def load_model_and_scaler():
    """
    Wczytuje wcześniej wytrenowany model KNN oraz scaler.
    """

    # Sprawdzamy, czy pliki modelu i scalera istnieją, przed próbą ich wczytania.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Nie znaleziono scalera: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Zwracamy gotowy do użycia model oraz scaler do transformacji danych.
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

    # Wczytujemy model i scaler z plików.
    model, scaler = load_model_and_scaler()

    # Analizujemy obraz, aby uzyskać wektor cech wejściowych.
    features = analyze_image(image_path)

    # Skalujemy cechy tym samym scalerem, który był użyty podczas treningu.
    features_scaled = scaler.transform([features])

    # Predykcja klasy liścia na podstawie przetworzonych cech.
    prediction = model.predict(features_scaled)[0]

    label = int(prediction)

    # Tłumaczymy etykietę na nazwę klasy, jeśli jest dostępna.
    class_name = CLASS_NAMES.get(label, f"Nieznana klasa: {label}")

    return label, class_name




 #Przykład zastosowania funkcji predict_leaf do klasyfikacji pojedynczego zdjęcia liścia.
 '''
 image_path = input("Podaj ścieżkę do zdjęcia liścia: ")
 label, class_name = predict_leaf(
     image_path,
     show_debug=True
 )
 print("\nWynik klasyfikacji:")
 print(f"Label: {label}")
 print(f"Nazwa liścia: {class_name}")
 '''