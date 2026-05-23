# Klasyfikacja gatunków liści/drzew na podstawie zdjęcia

## Członkowie grupy
- Agata Mróz
- Marcel Snopkowski
- Mateusz Rudziński
- Nikola Słupska


## Opis projektu
Projekt wykonuje analizę obrazu liścia i klasyfikuje go do jednej z klas drzew przy użyciu modelu KNN. Projekt obejmuje:
- generowanie i wczytywanie zbioru danych,
- trenowanie modelu KNN na cechach wyekstrahowanych z obrazów,
- zapisywanie modelu i scalera,
- przewidywanie klasy liścia dla nowych zdjęć.
- w projekcie wykorzystywany jest zbiór danych __Swedish Leaf Dataset__

## Instalacja zależności
1. Utwórz środowisko wirtualne (opcjonalnie):

```bash
python -m venv .venv
```

2. Aktywuj środowisko:


```bash
.venv\Scripts\Activate
```
3. Zainstaluj wymagane pakiety:

```bash
pip install -r requirements.txt
```

## Uruchomienie projektu
1. Należy utworzyć nowy plik python, np ``main.py`` w katalogu roboczym
``` python
from zdjecia import generate_dataset
from train import train_model
from predict import predict_leaf


#Przykładowy folder to przechowywania zdjęć
directory = "images_train"

#Utworzenie zbioru danych o ścieżce "dataset.csv" na podstawie zdjęć w folderze "directory"
generate_dataset(directory, "dataset_without_5_new")

#Uczenie klasyfikatora 
train_model("dataset_without_5_new.csv")

#Przewidywanie klasy liścia dla nowego zdjęcia
image_path = input("Podaj ścieżkę do zdjęcia liścia: ")
image_path = f"{image_path}"
label, class_name = predict_leaf(
    image_path,
    show_debug=True
)
print("\nWynik klasyfikacji:")
print(f"Label: {label}")
print(f"Nazwa liścia: {class_name}")
```

Z racji, że w repozytorium znajduje się już plik dataset_without_5_new.csv w ktorym znajdują się odpowiednie dane do ćwiczenia klasyfikatora, wykorzystywanie metody generate_dataset jest niepotrzebne w przypadku testowania programu 

Aby uruchomić projekt, wykonaj:

```bash
python main.py
```
