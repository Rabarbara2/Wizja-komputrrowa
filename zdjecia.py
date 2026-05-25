from pathlib import Path

import cv2
import numpy as np
import re

from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern

def get_label(file_path: str):
    file_name = Path(file_path).name
    match = re.search(r"l(\d+)nr", file_name)
    if match:
        return int(match.group(1))
    else:
        return 0

def analyze_image(image_path: str):

    img = cv2.imread(image_path)  # wczytanie obrazu

    if img is None:
        raise Exception("Nie udało się wczytać obrazu")

    # zmiana rozmiaru żeby było deczko szybciej
    h, w = img.shape[:2]
    scale = 512 / w
    img = cv2.resize(img, (512, int(h * scale)))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # zamiana koloru na RGB bo inaczej liście są niebieskie

    # GrabCut - znalezienie liścia na obrazku

    # przygotowanie tych tych
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # kwadrat (prostokąt wsumie) początkowy, wszystko poza nim to tło, w nim tło + obiekt
    h, w = img.shape[:2]
    rect = (10, 10, w - 20, h - 20)

    # 10 iteracji, stać mnie
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # poprawka żeby "chyba tło" uznać za tło i "chyba obiekt" za obiekt
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # nałożenie maski na obraz
    segmented = img_rgb * mask2[:, :, np.newaxis]

    # Kontur

    gray_mask = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)  # zamina obrazu z maską na szary

    # progowanie - wszystko co czarne jest czarne, wszystko co nie to jest białe
    _, thresh = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
    # zamknięcie żeby nie było dziur w liściu (szumy itp)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # tylko zewnętrzne (ignoruje dziury w liściu) kontury i skompresowane
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise Exception("Nie znaleziono żadnego konturu")

    contour = max(contours, key=cv2.contourArea)  # weź tylko największy kontur

    # Hu Moments (certified Hu moment)

    # policz momenty (opis kształtu: pole, środek ciężkości, symetria itp.) z konturu
    moments = cv2.moments(contour)
    # policz momenty Hu - lepsze bo niezależne od wielkości i obrotu obiektu (flatten żeby był ładny wektor)
    hu = cv2.HuMoments(moments).flatten()
    # ogarnij liczby żeby EjAj łatwiej łyknął
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    # znormalizuj Hu od 0 do 1 dla EjAja (wybredny się znalazł)
    #    hu = hu.reshape(1, -1)
    #    norm_hu = StandardScaler().fit_transform(hu)
    #    norm_hu = norm_hu.flatten()
    # LBP czyli tekstura

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # zamień obrazek na szary

    # zamień tło na czarne (na podst. maski)
    gray = gray.copy()
    gray[mask2 == 0] = 0

    # opis tekstury na podstawie sąsiedztwa 8 pikseli
    lbp = local_binary_pattern(gray, P=8, R=1, method="ror")

    # histogram. podziel na 16 binów żeby było mniej danych dla EjAjki i normalizacja
    hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-10)

    # wektorek cech - sklej wszystko co do tej pory wiemy o liściu
    features = np.concatenate([hu, hist])

    # rząd do dancych jaki to liść + jego cechy (trzeba odciąć jaki to liść do uczenia!)
    return {
        "features": np.abs(features),
        "img_rgb": img_rgb,
        "segmented": segmented,
        "thresh": thresh,
        "lbp": lbp,
        "hist": hist
    }
    #return features


def generate_dataset(directory_path: str, target_path: str):
    directory = Path(directory_path)
    image_paths = [
        path for path in directory.iterdir()
    ]
    for index, file_path in enumerate(image_paths):
        print(file_path)
        label = get_label(file_path)      #rodzaj liścia z nazwy pliku
        features = analyze_image(file_path)
        row = np.append(label, features)
        print(f"[{index + 1}/{len(image_paths)}] OK: {file_path} -> label={label}")

        # zapis 10 liczb po przecinku (wystarczy imo)
        with open(target_path + ".csv", "ab") as f:
            np.savetxt(f, [row], delimiter=",", fmt="%.10f")

def pokaz_obrazek(img_rgb,segmented, thresh, lbp, hist, predykcja):
    plt.figure(figsize=(14, 8))

    plt.suptitle(f"Gatunek drzewa: {predykcja}", fontsize=14)

    plt.subplot(2, 3, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Segmentacja (GrabCut)")
    plt.imshow(segmented)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Maska")
    plt.imshow(thresh, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("LBP")
    plt.imshow(lbp, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Histogram LBP")
    plt.bar(range(len(hist)), hist)

    plt.tight_layout()
    plt.show()

# # ładne obrazki dla umilenia czasu, ale tylko jeden liść
#
# plt.figure(figsize=(14, 8))
#
# plt.subplot(2, 3, 1)
# plt.title("Oryginalny obraz")
# plt.imshow(img_rgb)
# plt.axis("off")
#
# plt.subplot(2, 3, 2)
# plt.title("Segmentacja (GrabCut)")
# plt.imshow(segmented)
# plt.axis("off")
#
# plt.subplot(2, 3, 3)
# plt.title("Maska")
# plt.imshow(thresh, cmap="gray")
# plt.axis("off")
#
# plt.subplot(2, 3, 4)
# plt.title("LBP")
# plt.imshow(lbp, cmap="gray")
# plt.axis("off")
#
# plt.subplot(2, 3, 5)
# plt.title("Histogram LBP")
# plt.bar(range(len(hist)), hist)
#
# plt.tight_layout()
# plt.show()

