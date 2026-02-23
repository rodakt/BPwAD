# Biblioteki Pythona w analizie danych

## Tomasz Rodak

Wykład 2

---

Literatura:

- [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) Christopher M. Bishop, "Pattern Recognition and Machine Learning", 2006.
- [PML-1](https://probml.github.io/pml-book/) Kevin P. Murphy, "Probabilistic Machine Learning: An Introduction", 2022.
- [Dokumentacja scikit-learn](https://scikit-learn.org/stable/)
- [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html)

## scikit-learn — estymatory, transformatory, potoki

Biblioteka scikit-learn wprowadza następujące wysokopoziomowe pojęcia:
- **estymator** (*estimator*) — obiekt tworzący deterministyczną funkcję na podstawie danych zgodnie z reprezentowanym w nim modelem. Każdy estymator posiada metodę `fit()`.
- **predyktor** (*predictor*) — estymator posiadający metodę `predict()`. Służy do przewidywania wartości wyjściowych na podstawie nowych danych.
- **transformator** (*transformer*) — estymator posiadający metodę `transform()`. Służy do przekształcania danych wejściowych na nową przestrzeń.
- **potok** (*pipeline*) — sekwencja transformatorów i estymatora końcowego, które są stosowane w kolejności.

### Estymatory, predyktory i transformatory

Estymator to obiekt reprezentujący model. Każdy estymator powinien posiadać metodę `fit()`, która przyjmuje dane i uczy model, czyli tworzy deterministyczną funkcję na podstawie danych i zgodnie z reprezentowanym w nim modelem. W zależności od swojej roli estymator posiada również metodę `predict()` (predyktor) lub `transform()` (transformator), a także ewentualnie `fit_predict()` lub `fit_transform()`.

Niektóre estymatory pełnią jednocześnie obie role — na przykład `KMeans` posiada zarówno `predict()`, jak i `transform()`.


#### Metoda `fit()`

Metoda `fit()` zwykle przyjmuje dwa argumenty:
- `X` — dane wejściowe (np. cechy, obrazy, teksty)
- `y` — dane wyjściowe (np. etykiety, wartości docelowe)

Metoda `fit()` modyfikuje stan estymatora „w miejscu" oraz zwraca dopasowany estymator — pozwala to na łączenie wielu wywołań w potoku.

Jeśli uczenie jest nienadzorowane, to argument `y` nie jest wymagany, zwykle automatycznie ustawiany jest na wartość `None`.

Częstym efektem wywołania metody `fit()` jest utworzenie atrybutów w estymatorze o nazwach kończących się na `_` (np. `coef_`, `intercept_`, `n_features_in_`, `n_classes_`), które przechowują różne informacje o stanie estymatora po wywołaniu metody `fit()`.

#### Metoda `predict()`

Metoda `predict()` przyjmuje dane wejściowe `X` i zwraca przewidywane wartości wyjściowe `y`. Jest to metoda, która jest wywoływana po `fit()` — model musi być najpierw wyuczony na danych, aby można było przewidywać nowe wartości.

#### Metoda `transform()`

Metoda `transform()` przyjmuje dane wejściowe `X` i zwraca ich przekształcenie na jakąś nową przestrzeń. Zwrócone dane oznacza się często jako `Xt`. Jest to metoda, która również jest wywoływana po `fit()`.

#### Metody `fit_transform()` i `fit_predict()`

Metody `fit_transform()` i `fit_predict()` są skrótami dla wywołania `fit()` i następnie `transform()` lub `predict()`.
Metoda `fit_transform()` przyjmuje dane wejściowe `X`, uczy estymator na tych danych wywołując `fit()` i zwraca przekształcone dane wyjściowe `Xt` wywołując `transform()`. Podobnie `fit_predict()` przyjmuje dane wejściowe `X`, uczy estymator na tych danych wywołując `fit()` i zwraca przewidywane wartości wyjściowe `y` wywołując `predict()`.

### Przykłady estymatorów wbudowanych w scikit-learn

#### Regresja logistyczna

Model regresji logistycznej jest implementowany w `sklearn.linear_model.LogisticRegression`. Jest to przykład predyktora uczenia nadzorowanego. Spośród metod opisanych powyżej posiada:
- `fit(X, y)` — uczy model na danych `X` i etykietach `y`. Po wywołaniu tej metody w estymatorze są dostępne atrybuty:
  - `coef_` — współczynniki regresji
  - `intercept_` — wyraz wolny
  - `n_features_in_` — liczba cech w danych wejściowych
  - `feature_names_in_` — nazwy cech w danych wejściowych
  - `classes_` — etykiety klas
  - `n_iter_` — liczba iteracji w procesie uczenia
- `predict(X)` — przewiduje etykiety klas dla nowych danych `X`

Dodatkowo, model ten, tak jak i wiele innych modeli szacujących prawdopodobieństwo przynależności do klas, posiada metodę `predict_proba(X)`, która zwraca prawdopodobieństwa przynależności do klas dla danych `X`.

#### K-means

Model K-średnich (*K-means*) jest implementowany w `sklearn.cluster.KMeans`. Jest to przykład estymatora uczenia nienadzorowanego, który pełni jednocześnie rolę predyktora i transformatora. Posiada wszystkie metody opisane powyżej:
- `fit(X)` — uczy model na danych `X`. Po wywołaniu tej metody w estymatorze są dostępne atrybuty:
  - `cluster_centers_` — współrzędne centroidów klastrów
  - `labels_` — etykiety klastrów dla każdego punktu danych
  - `inertia_` — suma kwadratów odległości punktów danych do ich centroidów
- `predict(X)` — przewiduje etykiety klastrów dla nowych danych `X`
- `fit_predict(X)` — uczy model na danych `X` i zwraca etykiety klastrów dla tych danych
- `transform(X)` — przekształca dane `X` na przestrzeń odległości do centroidów klastrów
- `fit_transform(X)` — uczy model na danych `X` i zwraca `X` przekształcone na przestrzeń odległości do centroidów klastrów.

#### PCA

Model analizy głównych składowych (*PCA*) jest implementowany w `sklearn.decomposition.PCA`. Jest to przykład transformatora uczenia nienadzorowanego. Spośród metod opisanych powyżej posiada:
- `fit(X)` — uczy model na danych `X`. Po wywołaniu tej metody w estymatorze dostępne są m.in. atrybuty:
  - `components_` — macierz głównych składowych
  - `explained_variance_` — wariancja wyjaśniona przez każdą główną składową
  - `singular_values_` — wartości osobliwe
- `transform(X)` — przekształca dane `X` na przestrzeń głównych składowych
- `fit_transform(X)` — uczy model na danych `X` i zwraca przekształcone dane `X` w przestrzeni głównych składowych

Dodatkowo, model ten, tak jak i wiele innych modeli transformujących dane, posiada metodę `inverse_transform(Xt)`, która przekształca dane `Xt` z powrotem do oryginalnej przestrzeni. Ponieważ PCA nie jest transformatorem różnowartościowym (podczas transformacji zmienia się liczba cech i część informacji jest tracona), to metoda `inverse_transform(Xt)` nie zwraca danych oryginalnych, a jedynie ich przybliżenie.

#### `MinMaxScaler`

Model `MinMaxScaler` jest implementowany w `sklearn.preprocessing.MinMaxScaler`. Jest to przykład nienadzorowanego transformatora. Spośród metod opisanych powyżej posiada:
- `fit(X)` — uczy model na danych `X`. Po wywołaniu tej metody w estymatorze dostępne są m.in. atrybuty:
  - `data_max_` — maksymalne wartości cech w danych
  - `data_min_` — minimalne wartości cech w danych
  - `data_range_` — zakres wartości cech w danych
  - `n_features_in_` — liczba cech w danych wejściowych
  - ...
- `transform(X)` — przekształca dane `X` na przestrzeń znormalizowaną do przedziału [0, 1]
- `fit_transform(X)` — uczy model na danych `X` i zwraca przekształcone dane `X` w przedziale [0, 1]

Dodatkowo, model ten, tak jak i wiele innych modeli transformujących dane, posiada metodę `inverse_transform(Xt)`, która przekształca dane `Xt` z powrotem do oryginalnej przestrzeni.

#### `PolynomialFeatures`

Model `PolynomialFeatures` jest implementowany w `sklearn.preprocessing.PolynomialFeatures`. Jest to przykład nienadzorowanego transformatora służącego do generowania cech wielomianowych. Spośród metod opisanych powyżej posiada:
- `fit(X)` — uczy model na danych `X`. Po wywołaniu tej metody w estymatorze dostępne są m.in. atrybuty:
  - `powers_` — wykładniki cech wejściowych w wyjściu
  - `n_features_in_` — liczba cech w danych wejściowych
  - `n_output_features_` — liczba cech wyjściowych
  - ...
- `transform(X)` — przekształca dane `X` na przestrzeń cech wielomianowych
- `fit_transform(X)` — uczy model na danych `X` i zwraca przekształcone dane `X` w przestrzeni cech wielomianowych

### Typowy schemat wykorzystania estymatora

1. Inicjalizacja. Polega na stworzeniu obiektu estymatora, który reprezentuje model. Parametry estymatora są hiperparametrami modelu i są ustalane przed rozpoczęciem uczenia.
2. Uczenie. Polega na wywołaniu metody `fit()` na obiekcie estymatora, co prowadzi do dopasowania modelu do danych.
3. Przewidywanie lub transformacja. Polega na wywołaniu metody `predict()` lub `transform()` na obiekcie estymatora, co prowadzi do uzyskania przewidywanych wartości wyjściowych lub przekształconych danych wejściowych.

### Tworzenie własnych estymatorów w scikit-learn

Tworzenie własnych estymatorów w scikit-learn wymaga zrozumienia interfejsu API biblioteki oraz konwencji, które ściśle definiują, jak powinien zachowywać się estymator.

#### Podstawowe zasady

1. **Dziedziczenie po odpowiednich klasach bazowych**:
   - `BaseEstimator` — zapewnia funkcje pomocnicze, np. `get_params()` / `set_params()`
   - Odpowiedni *mixin* w zależności od typu estymatora:
     - `ClassifierMixin` — dla klasyfikatorów (dodaje domyślny `score()` = accuracy)
     - `RegressorMixin` — dla algorytmów regresji (dodaje domyślny `score()` = R²)
     - `TransformerMixin` — dla transformatorów danych (dodaje domyślny `fit_transform()`)
     - `ClusterMixin` — dla algorytmów klasteryzacji (dodaje domyślny `fit_predict()`)

2. **Konwencje nazewnictwa**:
   - Hiperparametry jako atrybuty inicjalizacji (np. `self.param = param` w `__init__`)
   - Atrybuty wyuczone kończą się podkreślnikiem (np. `self.coef_`)
   - Metoda `fit()` zwraca `self` dla umożliwienia łańcuchowania

3. **Walidacja danych** za pomocą funkcji z `sklearn.utils.validation`:
   - `check_X_y(X, y)` — sprawdza dane wejściowe w metodzie `fit()`
   - `check_array(X)` — sprawdza dane w metodzie `predict()` / `transform()`
   - `check_is_fitted(self)` — sprawdza, czy model został wytrenowany

Prawidłowo zaimplementowany estymator można używać w potokach (`Pipeline`), walidacji krzyżowej (`cross_val_score`) i przeszukiwaniu hiperparametrów (`GridSearchCV`).

#### Przykład: własny regresor

Klasa `RegresjaLiniowa` jest przykładem własnego estymatora regresji liniowej zgodnego z API scikit-learn. Model wyznacza współczynniki metodą najmniejszych kwadratów (równanie normalne), bez iteracyjnej optymalizacji.

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class RegresjaLiniowa(BaseEstimator, RegressorMixin):
    """Regresja liniowa wyznaczana metodą najmniejszych kwadratów."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_b = np.c_[np.ones(X.shape[0]), X]
        else:
            X_b = X

        # Równanie normalne: β = (X^T X)^{-1} X^T y
        # W praktyce korzystamy z pseudoodwrotności (lstsq),
        # która działa również dla macierzy osobliwych.
        beta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_ + self.intercept_
```

Porównanie z wbudowaną implementacją scikit-learn:

```python
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100) * 0.5

# Nasz estymator
model_wlasny = RegresjaLiniowa()
model_wlasny.fit(X, y)
print("R² (własny):", model_wlasny.score(X, y))

# Estymator scikit-learn
from sklearn.linear_model import LinearRegression
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)
print("R² (sklearn):", model_sklearn.score(X, y))
```

Metoda `score()` nie jest zdefiniowana w klasie `RegresjaLiniowa` — jest dostarczana automatycznie przez `RegressorMixin` (zwraca współczynnik determinacji R²).

#### Przykład: własny transformator

Poniższy transformator standaryzuje dane (odejmuje średnią, dzieli przez odchylenie standardowe). Ilustruje wzorzec `fit()` → `transform()` → `inverse_transform()`:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class Standaryzator(BaseEstimator, TransformerMixin):
    """Standaryzacja cech: (X - średnia) / odchylenie."""

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, Xt):
        check_is_fitted(self)
        Xt = check_array(Xt)
        return Xt * self.scale_ + self.mean_
```

Dzięki `TransformerMixin` klasa automatycznie zyskuje metodę `fit_transform()`. Transformator ten można od razu użyć w potoku — co pokazują kolejne sekcje.


## Potoki (*Pipeline*)

Potoki to mechanizm, który umożliwia łączenie wielu kroków przetwarzania danych w jedną spójną jednostkę.

### Podstawowa idea

Potok w scikit-learn pozwala na sekwencyjne łączenie różnych transformatorów i jednego estymatora końcowego w jedną całość. Zapewnia to:

1. **Spójny interfejs** — cały potok zachowuje się jak pojedynczy estymator scikit-learn z metodami `fit()`, `predict()`, itd.
2. **Automatyczne przekazywanie danych** — dane są automatycznie przekazywane między etapami.
3. **Eliminację wycieków danych** — potok zapewnia, że transformacje (np. skalowanie) są dopasowywane tylko do danych treningowych.

### Tworzenie potoku

Podstawowy potok tworzy się przy użyciu klasy `Pipeline`. Każdy krok to para `(nazwa, estymator)`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Cały potok można trenować jak pojedynczy estymator
pipe.fit(X_train, y_train)

# I używać do predykcji
predictions = pipe.predict(X_test)
```

Wywołanie `pipe.fit(X_train, y_train)` powoduje sekwencyjne wykonanie: `scaler.fit_transform(X_train)` → `classifier.fit(X_transformed, y_train)`. Analogicznie `pipe.predict(X_test)` wykonuje: `scaler.transform(X_test)` → `classifier.predict(X_transformed)`. Kluczowe jest to, że na danych testowych wywoływana jest metoda `transform()`, a nie `fit_transform()` — dzięki temu scaler nie „widzi" danych testowych.

### Główne zalety potoków

#### Organizacja kodu

Potoki organizują kod w zrozumiały i łatwy do utrzymania sposób — ważne, gdy łańcuch przetwarzania jest złożony.

#### Zapobieganie wyciekom danych

Jednym z najważniejszych zastosowań potoków jest zapobieganie wyciekom danych podczas preprocesingu.

Niepoprawne podejście (wyciek informacji):

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Skalowanie przed kroswalidacją - BŁĄD!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Wyciek: scaler widzi cały zbiór

# Kroswalidacja
model = LinearRegression()
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
```

W tym podejściu `StandardScaler` oblicza średnią i odchylenie standardowe na całym zbiorze danych, w tym na danych walidacyjnych. To zawyża wyniki kroswalidacji, bo model pośrednio „zna" statystyki danych, na których jest testowany.

Poprawne podejście z potokiem:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Kroswalidacja - skalowanie odbywa się osobno w każdym foldzie!
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')

print("RMSE dla każdego folda:", -scores)
print("Średni RMSE:", -scores.mean())
```

#### Optymalizacja hiperparametrów

Potoki są szczególnie przydatne w połączeniu z przeszukiwaniem hiperparametrów. Parametry poszczególnych kroków potoku adresuje się w formacie `nazwa_kroku__nazwa_parametru` (podwójne podkreślenie).

**`GridSearchCV`** przeszukuje wszystkie kombinacje wartości z podanej siatki:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'scaler__with_mean': [True, False],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l1', 'l2']
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
search.fit(X_train, y_train)

print("Najlepsze parametry:", search.best_params_)
print("Najlepszy wynik:", search.best_score_)
```

W powyższym przykładzie `GridSearchCV` przeszukuje 2 × 3 × 2 = 12 kombinacji, z pięciokrotną kroswalidacją — łącznie 60 dopasowań modelu. Przy większej liczbie hiperparametrów i ich wartości liczba kombinacji rośnie wykładniczo.

**`RandomizedSearchCV`** losuje zadaną liczbę kombinacji zamiast przeszukiwać wszystkie. Jest znacznie wydajniejszy przy dużej przestrzeni hiperparametrów:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'classifier__C': uniform(0.01, 100),         # rozkład ciągły
    'classifier__max_iter': randint(100, 1000),   # rozkład dyskretny
}

search = RandomizedSearchCV(
    pipe, param_distributions,
    n_iter=20,   # liczba losowanych kombinacji
    cv=5,
    scoring='accuracy',
    random_state=42
)
search.fit(X_train, y_train)
```

Oba narzędzia mają ograniczenia: `GridSearchCV` nie skaluje się do dużych przestrzeni, a `RandomizedSearchCV` nie uczy się z poprzednich prób. Bardziej zaawansowane podejścia — np. optymalizacja bayesowska — zostaną omówione w kolejnym wykładzie (Optuna).

#### Serializacja i wdrażanie

Cały potok można zapisać jako jeden obiekt:

```python
from joblib import dump, load

# Zapisz cały potok
dump(pipe, 'model_pipeline.joblib')

# Załaduj go później
loaded_pipe = load('model_pipeline.joblib')
```

Serializacja obejmuje zarówno wyuczone parametry estymatora, jak i stany wszystkich transformatorów — dzięki temu wdrożenie modelu polega na załadowaniu jednego pliku i wywołaniu `predict()`.

### Zaawansowane funkcje potoków

#### `ColumnTransformer`

W praktyce dane zawierają kolumny różnych typów (numeryczne, kategoryczne), które wymagają różnych transformacji. `ColumnTransformer` pozwala przypisać osobny transformator do wybranych kolumn:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['wiek', 'dochod']),
        ('cat', OneHotEncoder(), ['kategoria', 'region'])
    ],
    remainder='passthrough'  # kolumny nieuwzględnione powyżej — przekaż bez zmian
    # remainder='drop'       # alternatywnie: pomiń nieuwzględnione kolumny
)

pipe = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])
```

Parametr `remainder` kontroluje, co dzieje się z kolumnami, które nie zostały przypisane do żadnego transformatora. Domyślnie jest ustawiony na `'drop'` (pominięcie), co może prowadzić do trudnych do wychwycenia błędów, gdy dodamy nową kolumnę do danych i zapomnimy zaktualizować `ColumnTransformer`.

#### `FeatureUnion`

`FeatureUnion` pozwala na równoległe przetwarzanie danych przez różne transformatory i konkatenację wyników. Przydaje się, gdy chcemy wzbogacić dane o cechy uzyskane różnymi metodami:

```python
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

features = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('select_best', SelectKBest(k=2))
])

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('features', features),
    ('classifier', LogisticRegression())
])
```

W tym przykładzie dane po skalowaniu trafiają równolegle do PCA (redukcja wymiarów) i `SelectKBest` (selekcja cech). Wyniki obu transformatorów są łączone (horyzontalnie) i przekazywane do klasyfikatora. Końcowa macierz cech ma 2 + 2 = 4 kolumny.

Różnica między `ColumnTransformer` a `FeatureUnion`: `ColumnTransformer` dzieli dane *po kolumnach* (różne transformacje dla różnych cech), natomiast `FeatureUnion` przekazuje *te same dane* do wielu transformatorów i łączy wyniki.