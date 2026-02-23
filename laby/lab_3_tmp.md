# Biblioteki Pythona w analizie danych
## Tomasz Rodak

Lab 3

---

W tym arkuszu zaimplementujemy algorytm k-średnich od podstaw, korzystając wyłącznie z NumPy. Następnie zastosujemy go do segmentacji obrazów. Ćwiczenie łączy broadcasting, indeksowanie złożone, agregacje z `axis` i operacje na tablicach 3D.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```

## 1. Algorytm k-średnich

Algorytm k-średnich (*k-means*) dzieli zbiór $N$ punktów w $\mathbb{R}^d$ na $k$ grup (klastrów) tak, aby punkty wewnątrz każdej grupy były jak najbliżej środka (centroidu) swojej grupy. Jest to algorytm uczenia nienadzorowanego — nie korzysta z etykiet.

### Pseudokod

**Wejście:** macierz danych $X$ o kształcie $(N, d)$, liczba klastrów $k$, maksymalna liczba iteracji $I_{\max}$.

1. **Inicjalizacja:** Wylosuj $k$ różnych punktów z $X$ jako początkowe centroidy $C$ o kształcie $(k, d)$.
2. **Powtarzaj** (do zbieżności lub $I_{\max}$ iteracji):
   1. **Przypisanie:** Dla każdego punktu $x_i$ wyznacz najbliższy centroid:
   $$\text{labels}[i] = \arg\min_{j=1,\ldots,k} \|x_i - c_j\|^2$$
   2. **Aktualizacja:** Oblicz nowe centroidy jako średnie punktów przypisanych do każdego klastra:
   $$c_j = \frac{1}{|S_j|}\sum_{x_i \in S_j} x_i, \qquad S_j = \{x_i : \text{labels}[i] = j\}$$
   3. **Sprawdzenie zbieżności:** Jeśli centroidy nie zmieniły się (lub zmieniły się o mniej niż zadana tolerancja $\varepsilon$), zakończ.

**Wyjście:** wektor etykiet `labels` o kształcie $(N,)$, macierz centroidów $C$ o kształcie $(k, d)$.


## 2. Dane testowe

Do testowania implementacji użyjemy funkcji `make_blobs()` z biblioteki scikit-learn, która generuje grupy punktów na płaszczyźnie:

```python
from sklearn.datasets import make_blobs

N = 500
d = 2
k = 5

X, y_true = make_blobs(n_samples=N, centers=k, n_features=d, random_state=1)
plt.scatter(X[:, 0], X[:, 1], s=15)
plt.title('Dane testowe')
plt.axis('equal');
```

Wektor `y_true` zawiera prawdziwe etykiety — posłuży do oceny wyników, ale algorytm k-średnich go nie otrzymuje.

```python

```

## 3. Implementacja krok po kroku

### 3.1 Inicjalizacja centroidów

Wylosuj $k$ różnych indeksów ze zbioru $\{0, 1, \ldots, N-1\}$ i wybierz odpowiadające im punkty z $X$ jako początkowe centroidy.

*Wskazówka:* `rng.choice(N, size=k, replace=False)`, gdzie `rng = np.random.default_rng(seed=42)`.

Jaki jest kształt tablicy centroidów?

```python

```

### 3.2 Obliczenie odległości (broadcasting)

Oblicz kwadrat odległości euklidesowej między każdym punktem a każdym centroidem. Wynikiem powinna być tablica o kształcie $(N, k)$.

Przeanalizuj transformację kształtów:
```
X:          (N, d)  →  (N, 1, d)
C:          (k, d)  →  (1, k, d)
różnica:               (N, k, d)
kwadrat + suma po osi 2:  (N, k)
```

Zaimplementuj to w jednej linii, używając `np.newaxis` (lub `None`) i `np.sum()` z parametrem `axis`.

```python

```

Sprawdź wynik na małym przykładzie, np.:
```python
X_test = np.array([[0, 0], [1, 1], [5, 5]])   # (3, 2)
C_test = np.array([[0, 0], [5, 5]])             # (2, 2)
# Oczekiwane odległości^2: [[0, 50], [2, 32], [50, 0]]
```

```python

```

### 3.3 Przypisanie etykiet

Na podstawie tablicy odległości z kroku 3.2, przypisz każdemu punktowi etykietę najbliższego centroidu.

*Wskazówka:* `np.argmin()` z odpowiednim `axis`.

Jaki jest kształt wynikowego wektora etykiet?

```python

```

### 3.4 Aktualizacja centroidów

Oblicz nowe centroidy jako średnie współrzędne punktów przypisanych do każdego klastra.

*Wskazówka:* Dla klastra $j$ maska `labels == j` daje tablicę boolowską, którą można użyć do wybrania odpowiednich wierszy z $X$. Następnie `np.mean(..., axis=0)` daje nowy centroid.

Napisz pętlę po $j \in \{0, \ldots, k-1\}$, która buduje nową macierz centroidów. Zwróć uwagę na przypadek brzegowy: co jeśli żaden punkt nie został przypisany do klastra $j$? (W praktyce zdarza się to rzadko, ale warto obsłużyć — np. zachować stary centroid.)

```python

```

### 3.5 Złożenie algorytmu

Połącz kroki 3.1–3.4 w funkcję:

```python
def kmeans(X, k, max_iter=100, tol=1e-6, seed=42):
    """
    Algorytm k-średnich.
    
    Parametry
    ---------
    X : ndarray, kształt (N, d)
        Dane wejściowe.
    k : int
        Liczba klastrów.
    max_iter : int
        Maksymalna liczba iteracji.
    tol : float
        Tolerancja zbieżności (maksymalna zmiana centroidu).
    seed : int
        Ziarno generatora losowego.
    
    Zwraca
    ------
    labels : ndarray, kształt (N,)
        Etykiety klastrów.
    centroids : ndarray, kształt (k, d)
        Końcowe centroidy.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Inicjalizacja
    # ...
    
    for i in range(max_iter):
        # 2a. Oblicz odległości
        # ...
        
        # 2b. Przypisz etykiety
        # ...
        
        # 2c. Oblicz nowe centroidy
        # ...
        
        # 2d. Sprawdź zbieżność
        # ...
        pass
    
    return labels, centroids
```

Uzupełnij ciało funkcji, wykorzystując kod z kroków 3.2–3.4.

Kryterium zbieżności: oblicz maksymalną zmianę pozycji centroidu (`np.max(np.linalg.norm(new_centroids - centroids, axis=1))`) i przerwij pętlę, jeśli jest mniejsza niż `tol`.

```python

```


## 4. Testy na danych syntetycznych

### 4.1 Wizualizacja wyników

Uruchom swoją implementację na danych z sekcji 2. Narysuj wynik: punkty pokolorowane według przypisanych etykiet oraz centroidy oznaczone wyraźnym markerem (np. `plt.scatter(..., marker='x', s=200, c='red')`).

```python

```

### 4.2 Porównanie z sklearn

Uruchom `sklearn.cluster.KMeans` na tych samych danych i porównaj wyniki:

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=k, n_init=10, random_state=42)
km.fit(X)
```

1. Porównaj centroidy: `km.cluster_centers_` vs twoje centroidy. Czy są zbliżone? (Etykiety mogą być permutacją — klaster 0 u ciebie może odpowiadać klastrowi 3 w sklearn.)
2. Porównaj wartość funkcji celu (*inertia*): $\sum_{i=1}^{N} \|x_i - c_{\text{labels}[i]}\|^2$. Oblicz ją dla swoich wyników i porównaj z `km.inertia_`.

```python

```

### 4.3 Wpływ inicjalizacji

Uruchom swoją implementację kilka razy z różnymi wartościami `seed`. Czy wyniki zawsze są takie same? Jak zmienia się *inertia*?

Sklearn domyślnie uruchamia algorytm `n_init=10` razy i wybiera najlepszy wynik. Zaimplementuj analogiczną strategię: uruchom `kmeans()` dla kilku ziaren i wybierz wynik o najniższej wartości funkcji celu.

```python

```


## 5. Segmentacja obrazów

Segmentacja obrazu polega na podziale pikseli na grupy o podobnych kolorach. Algorytm k-średnich nadaje się do tego idealnie: traktujemy wartości RGB każdego piksela jako punkt w przestrzeni $\mathbb{R}^3$ i grupujemy je w $k$ klastrów. Każdy piksel zastępujemy kolorem centroidu jego klastra.

### 5.1 Wczytanie obrazu

Pobierz dowolny kolorowy obraz RGB z internetu i wczytaj go jako tablicę NumPy. Możesz skorzystać z biblioteki `requests`:

```python
import requests

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Common_brimstone_%28Gonepteryx_rhamni%29_female_underside.JPG/1280px-Common_brimstone_%28Gonepteryx_rhamni%29_female_underside.JPG'
r = requests.get(url)
with open('motyl.jpg', 'wb') as f:
    f.write(r.content)
```

```python

```

### 5.2 Przygotowanie danych

Przekształć tablicę obrazu o kształcie $(H, W, 3)$ na tablicę 2D o kształcie $(N, 3)$, gdzie $N = H \times W$.

Algorytm k-średnich operuje na liczbach zmiennoprzecinkowych, a piksele są typu `uint8`. Przekształć tablicę do typu `float64`.

Zapamiętaj oryginalny kształt obrazu — będzie potrzebny do odtworzenia obrazu wynikowego.

```python

```

### 5.3 Segmentacja

Uruchom swoją implementację `kmeans()` na przygotowanej tablicy pikseli. Zacznij od $k = 3$.

**Uwaga dotycząca wydajności:** Obraz może mieć setki tysięcy pikseli. Jeśli algorytm działa wolno, możesz:
* zmniejszyć obraz przed segmentacją (np. `img_array[::2, ::2]`),
* ograniczyć `max_iter`.

```python

```

### 5.4 Rekonstrukcja obrazu

1. Utwórz tablicę wynikową, w której każdy piksel ma wartości RGB centroidu swojego klastra. Wykorzystaj indeksowanie złożone: jeśli `labels` to wektor etykiet, a `centroids` to macierz centroidów, to `centroids[labels]` daje tablicę $(N, 3)$, w której $i$-ty wiersz to centroid klastra, do którego należy $i$-ty piksel.
2. Przekształć wynik do typu `uint8` (metoda `.astype(np.uint8)`).
3. Przywróć oryginalny kształt (`reshape()`).
4. Wyświetl obraz za pomocą `Image.fromarray()`.

```python

```

### 5.5 Eksperymenty

1. Przetestuj segmentację dla różnych wartości $k$: 2, 4, 8, 16. Wyświetl wyniki obok siebie.
2. Ile iteracji potrzebuje algorytm do zbieżności? Dodaj do swojej funkcji `kmeans()` zliczanie iteracji i wypisywanie wartości funkcji celu w każdej iteracji.
3. Porównaj wynik swojej implementacji z `sklearn.cluster.KMeans` na tym samym obrazie. Czy segmentacje wyglądają podobnie?

```python

```

```python

```

## 6. Zadanie dodatkowe: kompresja obrazu

Segmentacja k-średnich to de facto kompresja kolorów: zamiast $256^3 \approx 16{,}7$ mln możliwych kolorów, obraz używa tylko $k$. 

1. Oblicz współczynnik kompresji. Oryginalny obraz potrzebuje $H \times W \times 3$ bajtów. Obraz skompresowany potrzebuje: tablicę etykiet ($H \times W$ wartości, każda wymagająca $\lceil\log_2 k\rceil$ bitów) plus tablicę centroidów ($k \times 3$ bajtów). Jaki jest stosunek rozmiaru skompresowanego do oryginalnego dla $k = 4, 8, 16$?
2. Zbadaj zależność między $k$ a jakością wizualną. Jaka jest najmniejsza wartość $k$, przy której obraz wygląda akceptowalnie?

```python

```