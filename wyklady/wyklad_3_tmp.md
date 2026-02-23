# Biblioteki Pythona w analizie danych

## Tomasz Rodak

Wykład 3

---

Literatura:

- [Dokumentacja Optuna](https://optuna.readthedocs.io/en/stable/)
- [Dokumentacja optuna-integration](https://optuna-integration.readthedocs.io/en/latest/)
- [Dokumentacja scikit-learn](https://scikit-learn.org/stable/)
- [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) Christopher M. Bishop, "Pattern Recognition and Machine Learning", 2006.
- [PML-1](https://probml.github.io/pml-book/) Kevin P. Murphy, "Probabilistic Machine Learning: An Introduction", 2022.
- T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902), KDD 2019.

## Optuna i optymalizacja hiperparametrów

### Problem doboru hiperparametrów

Hiperparametry to parametry modelu ustalane przed rozpoczęciem uczenia — nie są wyznaczane przez algorytm treningowy, lecz wybierane przez użytkownika. W terminologii scikit-learn odpowiadają argumentom konstruktora `__init__`, w odróżnieniu od parametrów wyuczonych (atrybuty kończące się na `_`, np. `coef_`, `intercept_`).

Przykłady hiperparametrów:
- `C` w `LogisticRegression` — siła regularyzacji,
- `n_neighbors` w `KNeighborsClassifier` — liczba sąsiadów,
- `degree` w `PolynomialFeatures` — stopień wielomianu,
- `alpha` w `Ridge` / `Lasso` — współczynnik regularyzacji.

Dobór hiperparametrów wpływa bezpośrednio na jakość modelu. Ręczne próbowanie kilku wartości nie skaluje się — przy $k$ hiperparametrach, z których każdy może przyjmować $m$ wartości, przestrzeń przeszukiwania ma rozmiar $m^k$. Potrzebujemy metod systematycznego przeszukiwania tej przestrzeni.


### Przypomnienie: `GridSearchCV` i `RandomizedSearchCV`

Na poprzednim wykładzie poznaliśmy dwa narzędzia do optymalizacji hiperparametrów wbudowane w scikit-learn.

#### `GridSearchCV`

`GridSearchCV` przeszukuje wszystkie kombinacje wartości z podanej siatki. Dla potoku `StandardScaler` → `LogisticRegression` z trzema hiperparametrami (`with_mean`: 2 wartości, `C`: 3 wartości, `penalty`: 2 wartości) daje 2 × 3 × 2 = 12 kombinacji. Z pięciokrotną kroswalidacją to 60 dopasowań modelu.

Problem pojawia się przy większej liczbie hiperparametrów. Pięć hiperparametrów po 10 wartości daje $10^5 = 100\,000$ kombinacji — przy kroswalidacji z $k=5$ foldami to pół miliona dopasowań. Koszt rośnie wykładniczo.

#### `RandomizedSearchCV`

`RandomizedSearchCV` losuje zadaną liczbę kombinacji (`n_iter`) z podanych rozkładów. Dzięki temu koszt jest kontrolowany i nie zależy od rozmiaru siatki. Pozwala też na przeszukiwanie przestrzeni ciągłych (np. `C` z rozkładu `uniform(0.01, 100)`).

Ograniczenie jest jednak fundamentalne: każda próba jest niezależna od poprzednich. Sampler nie uczy się, które regiony przestrzeni dają lepsze wyniki. Dwudziesta próba jest równie „ślepa" jak pierwsza.

#### Podsumowanie ograniczeń

| Cecha | `GridSearchCV` | `RandomizedSearchCV` |
|---|---|---|
| Przeszukiwanie | wyczerpujące | losowe |
| Skalowalność | słaba (wzrost wykładniczy) | dobra (kontrolowana przez `n_iter`) |
| Adaptacyjność | brak | brak |
| Przestrzenie ciągłe | nie (wymaga dyskretyzacji) | tak |
| Wczesne zatrzymanie | nie | nie |

Oba narzędzia nie wykorzystują informacji z poprzednich prób do kierowania dalszym przeszukiwaniem. Potrzebujemy metody, która **uczy się** z dotychczasowych obserwacji — to właśnie realizuje optymalizacja bayesowska.


### Optymalizacja bayesowska — intuicja

Optymalizacja bayesowska to rodzina metod, które kierują przeszukiwaniem przestrzeni hiperparametrów na podstawie dotychczasowych wyników. Zamiast przeszukiwać ślepo, budują wewnętrzny model zależności między hiperparametrami a jakością modelu.

#### Model surogatowy

Ewaluacja funkcji celu (np. kroswalidacja modelu z danymi hiperparametrami) jest kosztowna. Optymalizacja bayesowska buduje tani *model surogatowy* (surrogate model), który przybliża nieznaną funkcję celu na podstawie dotychczasowych obserwacji. Model surogatowy pozwala szybko oszacować, jak dobrze sprawdzą się jeszcze niesprawdzone kombinacje hiperparametrów.

#### Funkcja akwizycji

Na podstawie modelu surogatowego konstruowana jest *funkcja akwizycji* (acquisition function), która decyduje, który punkt przestrzeni zbadać w kolejnym kroku. Funkcja akwizycji balansuje dwa cele:
- **eksploatacja** (*exploitation*) — przeszukiwanie regionów, które dotychczas dawały dobre wyniki,
- **eksploracja** (*exploration*) — próbowanie regionów, o których wiemy mało (wysoka niepewność).

#### TPE — Tree-structured Parzen Estimator

Domyślna metoda optymalizacji w Optunie to TPE. Zamiast modelować bezpośrednio zależność $f(\mathbf{x})$ (wynik od hiperparametrów), TPE modeluje dwa rozkłady warunkowe:
- $p(\mathbf{x} \mid y < y^*)$ — rozkład hiperparametrów w próbach „dobrych" (poniżej pewnego progu $y^*$),
- $p(\mathbf{x} \mid y \geq y^*)$ — rozkład hiperparametrów w próbach „słabych".

Następny punkt do zbadania wybierany jest tak, aby zmaksymalizować stosunek $p(\mathbf{x} \mid y < y^*) / p(\mathbf{x} \mid y \geq y^*)$ — preferowane są hiperparametry typowe dla dobrych prób, a nietypowe dla słabych.

Kluczowa konsekwencja: każda kolejna próba jest mądrzejsza od poprzedniej, bo korzysta z akumulowanej wiedzy o przestrzeni hiperparametrów.


### Optuna — podstawowe pojęcia

Optuna to biblioteka do optymalizacji hiperparametrów, która implementuje m.in. algorytm TPE. Została zaprojektowana zgodnie z paradygmatem *define-by-run*: przestrzeń przeszukiwania definiuje się wewnątrz funkcji celu, a nie w zewnętrznym słowniku. Daje to dużą elastyczność — przestrzeń może zależeć od wyborów dokonanych wcześniej w tej samej próbie.

Typowy import:

```python
import optuna
```

#### Study i Trial

Dwa centralne pojęcia Optuny:
- **Study** — obiekt zarządzający całym procesem optymalizacji. Przechowuje historię wszystkich prób i konfigurację (kierunek optymalizacji, sampler, pruner).
- **Trial** — pojedyncza próba, tj. jedna ewaluacja funkcji celu z konkretnymi wartościami hiperparametrów.

Tworzenie study:

```python
study = optuna.create_study(direction="minimize")
```

Argument `direction` przyjmuje wartość `"minimize"` (np. minimalizujemy MSE) lub `"maximize"` (np. maksymalizujemy accuracy). Domyślna wartość to `"minimize"`.

#### Funkcja celu

Funkcja celu (*objective*) to funkcja przyjmująca obiekt `trial` i zwracająca wartość liczbową do optymalizacji:

```python
def objective(trial):
    # ... budowa modelu, ewaluacja ...
    return score
```

Uruchomienie optymalizacji:

```python
study.optimize(objective, n_trials=100)
```

Metoda `optimize` wywołuje funkcję `objective` wielokrotnie (tu: 100 razy), za każdym razem przekazując nowy obiekt `trial` z hiperparametrami zasugerowanymi przez sampler.

#### Przestrzenie przeszukiwania

Wewnątrz funkcji celu hiperparametry pobieramy za pomocą metod `suggest_*` obiektu `trial`:

- `trial.suggest_float(name, low, high)` — wartość zmiennoprzecinkowa z przedziału $[\text{low}, \text{high}]$.
  - `log=True` — przeszukiwanie w skali logarytmicznej (przydatne np. dla learning rate, regularyzacji).
  - `step=0.1` — dyskretyzacja z zadanym krokiem.
- `trial.suggest_int(name, low, high)` — wartość całkowita z przedziału $[\text{low}, \text{high}]$.
  - `log=True` i `step` — działają analogicznie.
- `trial.suggest_categorical(name, choices)` — wybór z listy wartości.

Każdy hiperparametr identyfikowany jest przez unikalną nazwę (`name`). Optuna zapamiętuje te nazwy i wykorzystuje je w wizualizacjach oraz analizie ważności hiperparametrów.

Porównanie z podejściem scikit-learn:

| scikit-learn | Optuna |
|---|---|
| `param_grid = {'C': [0.1, 1, 10]}` | `trial.suggest_float('C', 0.1, 10, log=True)` |
| `param_distributions = {'C': uniform(0.01, 100)}` | `trial.suggest_float('C', 0.01, 100)` |
| Przestrzeń zdefiniowana w słowniku | Przestrzeń zdefiniowana w kodzie |
| Statyczna | Dynamiczna (może zależeć od innych wyborów) |


### Przykład: regresja wielomianowa z regularyzacją

Zbudujemy kompletny przykład optymalizacji hiperparametrów z użyciem Optuny. Problem: dopasowanie regresji wielomianowej z regularyzacją do danych syntetycznych, z wykorzystaniem potoku scikit-learn.

#### Dane

Generujemy dane z nieliniową zależnością i szumem:

```python
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 200
X = np.sort(np.random.uniform(-3, 3, n)).reshape(-1, 1)
y = np.sin(X.ravel()) + 0.3 * np.random.randn(n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Funkcja $\sin(x)$ z addytywnym szumem gaussowskim daje problem, w którym zbyt niski stopień wielomianu prowadzi do niedouczenia, a zbyt wysoki — do przeuczenia. Dobór hiperparametrów (stopień wielomianu, typ i siła regularyzacji) jest kluczowy.

#### Funkcja celu

Funkcja celu buduje potok `PolynomialFeatures` → `Ridge` lub `Lasso`, ewaluuje go kroswalidacją i zwraca średni błąd:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

def objective(trial):
    # Hiperparametry
    degree = trial.suggest_int("degree", 1, 15)
    alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
    model_name = trial.suggest_categorical("model", ["Ridge", "Lasso"])

    Model = Ridge if model_name == "Ridge" else Lasso

    # Potok
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", Model(alpha=alpha, max_iter=10000))
    ])

    # Ewaluacja: kroswalidacja, metryka = neg. MSE
    scores = cross_val_score(
        pipe, X_train, y_train,
        cv=5, scoring="neg_mean_squared_error"
    )
    return -scores.mean()  # minimalizujemy MSE
```

Zwróćmy uwagę na kilka elementów:
- `log=True` przy `alpha` — regularyzacja działa w różnych rzędach wielkości ($10^{-6}$ do $10^2$), więc skala logarytmiczna jest naturalna.
- `include_bias=False` w `PolynomialFeatures` — wyraz wolny doda sam model regresji.
- `scoring="neg_mean_squared_error"` — scikit-learn zwraca ujemny MSE (konwencja: wyżej = lepiej), więc negujemy wynik, aby Optuna minimalizowała właściwy MSE.

#### Optymalizacja

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

Domyślnie Optuna wypisuje logi z każdej próby. Można to wyciszyć:

```python
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

#### Odczyt wyników

```python
# Najlepsze hiperparametry
print(study.best_params)
# np. {'degree': 5, 'alpha': 0.032, 'model': 'Ridge'}

# Najlepsza wartość funkcji celu
print(study.best_value)
# np. 0.0893

# Pełne informacje o najlepszej próbie
print(study.best_trial)

# Historia wszystkich prób jako DataFrame
df = study.trials_dataframe()
df.head()
```

Metoda `study.trials_dataframe()` zwraca `pandas.DataFrame` z kolumnami zawierającymi numer próby, wartość celu, wartości hiperparametrów, czas trwania i stan próby.

#### Ewaluacja na zbiorze testowym

Po wybraniu najlepszych hiperparametrów trenujemy finalny model na całym zbiorze treningowym i oceniamy go na zbiorze testowym:

```python
from sklearn.metrics import mean_squared_error

best = study.best_params
Model = Ridge if best["model"] == "Ridge" else Lasso

final_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=best["degree"], include_bias=False)),
    ("reg", Model(alpha=best["alpha"], max_iter=10000))
])

final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"MSE na zbiorze testowym: {mse_test:.4f}")
```

Jest to standardowy schemat: kroswalidacja służy do wyboru hiperparametrów, a zbiór testowy — do ostatecznej, niezależnej oceny.


### Samplery

Sampler to algorytm decydujący, jakie wartości hiperparametrów zaproponować w kolejnej próbie. Optuna udostępnia kilka samplerów:

#### `TPESampler` (domyślny)

```python
study = optuna.create_study(sampler=optuna.samplers.TPESampler())
```

Tree-structured Parzen Estimator, opisany w sekcji o optymalizacji bayesowskiej. Adaptacyjny — uczy się z historii prób. Najlepszy wybór w typowych sytuacjach z umiarkowaną liczbą hiperparametrów (do kilkunastu).

Zachowanie na początku: TPE potrzebuje kilku losowych prób na rozgrzewkę (domyślnie `n_startup_trials=10`), zanim zacznie kierować przeszukiwaniem. Pierwsze próby są więc losowe.

#### `RandomSampler`

```python
study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
```

Odpowiednik `RandomizedSearchCV` — losuje hiperparametry niezależnie. Przydatny jako baseline do porównania z TPE lub przy bardzo dużych przestrzeniach, gdzie model surogatowy może być zawodny.

#### `GridSampler`

```python
search_space = {
    "degree": [1, 2, 3, 4, 5],
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
}
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
```

Odpowiednik `GridSearchCV` — przeszukuje wszystkie kombinacje z podanej siatki. Użyteczny, gdy przestrzeń jest mała i chcemy ją zbadać wyczerpująco, a jednocześnie chcemy korzystać z infrastruktury Optuny (wizualizacje, pruning, zapis wyników).


### Pruning — wczesne odrzucanie słabych prób

Pruning to mechanizm pozwalający na przerwanie ewaluacji próby, zanim zostanie ona w pełni zakończona. Jeśli po kilku krokach ewaluacji wynik jest wyraźnie gorszy od dotychczasowych prób, nie ma sensu kontynuować — oszczędzamy czas obliczeniowy.

W kontekście kroswalidacji „kroki" to poszczególne foldy: jeśli po trzech z pięciu foldów średni błąd jest znacznie gorszy od mediany zakończonych prób, kolejne dwa foldy raczej tego nie zmienią.

#### `MedianPruner`

Domyślny pruner Optuny. Odrzuca próbę w kroku $s$, jeśli jej najlepsza dotychczasowa wartość pośrednia jest gorsza od mediany wartości pośrednich w tym samym kroku $s$ wśród zakończonych prób.

```python
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
)
```

Parametry:
- `n_startup_trials` — liczba prób na początku, które nigdy nie są odrzucane (pruner potrzebuje danych referencyjnych).
- `n_warmup_steps` — liczba pierwszych kroków każdej próby, w których pruning jest wyłączony.

#### Implementacja pruningu z kroswalidacją

Aby korzystać z pruningu, musimy zastąpić `cross_val_score` ręczną pętlą po foldach. W każdym kroku raportujemy wynik pośredni i sprawdzamy, czy pruner zaleca przerwanie:

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def objective_with_pruning(trial):
    degree = trial.suggest_int("degree", 1, 15)
    alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)
    model_name = trial.suggest_categorical("model", ["Ridge", "Lasso"])

    Model = Ridge if model_name == "Ridge" else Lasso

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", Model(alpha=alpha, max_iter=10000))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for step, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        pipe.fit(X_train[train_idx], y_train[train_idx])
        y_val_pred = pipe.predict(X_train[val_idx])
        mse = mean_squared_error(y_train[val_idx], y_val_pred)
        fold_scores.append(mse)

        # Raportuj bieżącą średnią
        trial.report(np.mean(fold_scores), step)

        # Sprawdź, czy pruner zaleca przerwanie
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)
```

Kluczowe elementy:
- `trial.report(value, step)` — raportuje wartość pośrednią (średnią po dotychczasowych foldach) w danym kroku.
- `trial.should_prune()` — zwraca `True`, jeśli pruner uznaje próbę za mało obiecującą.
- `raise optuna.TrialPruned()` — przerywa próbę. Optuna oznacza ją jako odrzuconą (*pruned*), a nie zakończoną niepowodzeniem.

Uruchomienie:

```python
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
)
study.optimize(objective_with_pruning, n_trials=200)
```

Dzięki pruningowi możemy zwiększyć `n_trials` — wiele słabych prób zostanie przerwanych wcześnie, więc łączny koszt obliczeniowy nie rośnie proporcjonalnie.

#### Statystyki pruningu

Po zakończeniu optymalizacji możemy sprawdzić, ile prób zostało odrzuconych:

```python
pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Zakończone: {len(complete)}, odrzucone: {len(pruned)}")
```

#### Inne prunery

- `SuccessiveHalvingPruner` — inspirowany algorytmem Successive Halving; stopniowo eliminuje najsłabsze próby.
- `HyperbandPruner` — rozszerza Successive Halving o wiele „nawiasów" (brackets) z różnym budżetem początkowym.
- `PercentilePruner` — odrzuca próby poniżej zadanego percentyla (bardziej/mniej agresywny niż `MedianPruner`).


### `OptunaSearchCV` — integracja z scikit-learn

Dla użytkowników przyzwyczajonych do interfejsu `GridSearchCV` / `RandomizedSearchCV` dostępna jest klasa `OptunaSearchCV` z pakietu `optuna-integration`. Pozwala ona korzystać z algorytmów Optuny (TPE, pruning) w ramach interfejsu kompatybilnego z scikit-learn.

Instalacja:

```bash
pip install optuna-integration
```

Użycie:

```python
from optuna.distributions import IntDistribution, FloatDistribution
from optuna_integration.sklearn import OptunaSearchCV

pipe = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("reg", Ridge(max_iter=10000))
])

param_distributions = {
    "poly__degree": IntDistribution(1, 15),
    "reg__alpha": FloatDistribution(1e-6, 1e2, log=True),
}

search = OptunaSearchCV(
    pipe,
    param_distributions,
    cv=5,
    n_trials=100,
    scoring="neg_mean_squared_error",
    random_state=42
)

search.fit(X_train, y_train)

print("Najlepsze parametry:", search.best_params_)
print("Najlepszy wynik:", search.best_score_)
```

Uwagi:
- Nazwy hiperparametrów w `param_distributions` muszą być zgodne z konwencją potoków scikit-learn (`nazwa_kroku__nazwa_parametru`).
- Przestrzenie przeszukiwania definiuje się za pomocą klas z `optuna.distributions` (nie za pomocą `suggest_*`).
- `OptunaSearchCV` obsługuje `refit=True` (domyślnie), więc po zakończeniu optymalizacji najlepszy model jest automatycznie wytrenowany na pełnym zbiorze treningowym.
- Interfejs jest kompatybilny z resztą ekosystemu scikit-learn — obiekt `search` można traktować jak estymator (ma `predict()`, `score()` itd.).

#### Kiedy użyć `OptunaSearchCV`, a kiedy ręcznej funkcji celu?

`OptunaSearchCV` jest wygodny, gdy potrzebujemy prostego drop-in replacement dla `GridSearchCV`. Ręczna funkcja `objective` daje większą elastyczność:
- pruning z raportowaniem wyników pośrednich,
- warunkowe hiperparametry (np. `alpha` ma sens tylko gdy `model == "Ridge"`),
- optymalizacja modeli spoza scikit-learn (np. PyTorch, XGBoost),
- wielokryterialna optymalizacja.


### Warunkowe hiperparametry

Podejście *define-by-run* Optuny pozwala na definiowanie hiperparametrów, których obecność zależy od wartości innych hiperparametrów. Jest to trudne lub niemożliwe do wyrażenia w statycznym słowniku `param_grid`.

Przykład: jeśli wybieramy między `Ridge` a `Lasso`, to `Ridge` ma parametr `solver`, którego `Lasso` nie ma. Możemy to wyrazić naturalnie:

```python
def objective(trial):
    model_name = trial.suggest_categorical("model", ["Ridge", "Lasso"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e2, log=True)

    if model_name == "Ridge":
        solver = trial.suggest_categorical("ridge_solver", ["auto", "svd", "cholesky"])
        reg = Ridge(alpha=alpha, solver=solver)
    else:
        reg = Lasso(alpha=alpha, max_iter=10000)

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=trial.suggest_int("degree", 1, 15),
                                     include_bias=False)),
        ("reg", reg)
    ])

    scores = cross_val_score(
        pipe, X_train, y_train,
        cv=5, scoring="neg_mean_squared_error"
    )
    return -scores.mean()
```

Hiperparametr `ridge_solver` pojawia się tylko w próbach, w których `model == "Ridge"`. Optuna obsługuje to automatycznie — TPE modeluje warunkowe rozkłady, a wizualizacje prawidłowo uwzględniają brakujące wartości.


### Wizualizacja wyników

Moduł `optuna.visualization` udostępnia zestaw funkcji do analizy przebiegu i wyników optymalizacji. Wykresy oparte są na Plotly, więc są interaktywne.

#### Historia optymalizacji

```python
optuna.visualization.plot_optimization_history(study)
```

Wykres przedstawia wartość funkcji celu w funkcji numeru próby. Niebieskie punkty to poszczególne próby, czerwona linia to dotychczasowe najlepsze rozwiązanie. Pozwala ocenić, czy optymalizacja zbiegła — jeśli czerwona linia stabilizuje się, dalsze próby prawdopodobnie nie poprawią wyniku.

#### Ważność hiperparametrów

```python
optuna.visualization.plot_param_importances(study)
```

Wykres słupkowy przedstawiający względny wpływ każdego hiperparametru na wartość funkcji celu. Wewnętrznie korzysta z algorytmu fANOVA (functional ANOVA), który rozkłada wariancję funkcji celu na składowe związane z poszczególnymi hiperparametrami. Pozwala zidentyfikować, które hiperparametry warto optymalizować, a które mają marginalny wpływ.

#### Wykresy wycinkowe (slice plots)

```python
optuna.visualization.plot_slice(study)
```

Dla każdego hiperparametru rysowany jest osobny panel: wartość hiperparametru na osi $x$, wartość celu na osi $y$. Każdy punkt odpowiada jednej próbie. Pozwala dostrzec trendy — np. czy niskie wartości `alpha` konsekwentnie dają lepsze wyniki, albo czy istnieje optymalny zakres dla `degree`.

#### Mapa konturowa (contour plot)

```python
optuna.visualization.plot_contour(study, params=["degree", "alpha"])
```

Mapa cieplna zależności wartości celu od pary hiperparametrów. Pozwala wykryć interakcje — np. czy optymalny `degree` zależy od poziomu regularyzacji `alpha`.

#### Wykres współrzędnych równoległych

```python
optuna.visualization.plot_parallel_coordinate(study)
```

Każda próba to linia łącząca wartości poszczególnych hiperparametrów i wartość celu. Próby z niską wartością celu można podświetlić, co pozwala wizualnie zidentyfikować wzorce w najlepszych rozwiązaniach.

#### Plotly i Matplotlib

Domyślnie wykresy są generowane w Plotly. Alternatywnie dostępny jest backend Matplotlib:

```python
from optuna.visualization.matplotlib import plot_optimization_history
plot_optimization_history(study)
```


### Podsumowanie

| Cecha | `GridSearchCV` | `RandomizedSearchCV` | Optuna |
|---|---|---|---|
| Przeszukiwanie | wyczerpujące | losowe | adaptacyjne (TPE) |
| Skalowalność | słaba | dobra | dobra |
| Adaptacyjność | brak | brak | tak |
| Pruning | nie | nie | tak |
| Warunkowe hiperparametry | nie | nie | tak |
| Wizualizacja | brak | brak | tak (`optuna.visualization`) |
| Integracja z scikit-learn | natywna | natywna | `OptunaSearchCV` |
| Define-by-run | nie | nie | tak |

Optuna będzie wykorzystywana w kolejnych częściach kursu — w szczególności przy optymalizacji hiperparametrów sieci neuronowych w PyTorch, gdzie pruning na podstawie pośrednich wyników z kolejnych epok treningowych okaże się szczególnie przydatny.