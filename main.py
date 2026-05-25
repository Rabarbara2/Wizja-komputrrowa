from zdjecia import generate_dataset
from train import train_model
from predict import predict_leaf


#Przykładowy folder to przechowywania zdjęć
#directory = "images_train"

#Utworzenie zbioru danych o ścieżce "dataset.csv" na podstawie zdjęć w folderze "directory"
#generate_dataset(directory, "dataset_without_5_new")

#Uczenie klasyfikatora
train_model("dataset_without_5_new.csv")

#Przewidywanie klasy liścia dla nowego zdjęcia
image_path = input("Podaj ścieżkę do zdjęcia liścia: ")
label, class_name = predict_leaf(
    image_path,
    show_debug=True
)
print("\nWynik klasyfikacji:")
print(f"Label: {label}")
print(f"Nazwa liścia: {class_name}")