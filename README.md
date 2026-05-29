# Systémová príručka

Tento projekt obsahuje Python skript vytvorený pre praktickú časť bakalárskej práce zameranej na prediktívnu analýzu odchodovosti pacientov zo sledovania liečby pri chronických ochoreniach.

Slúži na popis a spustenie všetkých kódov.

Program načíta pripravený dataset, vytvorí odvodené premenné, rozdelí vstupné údaje na skupinu bez behaviorálnych premenných a skupinu s behaviorálnymi premennými, natrénuje viaceré klasifikačné modely a vyhodnotí ich pomocou vybraných metrík.

## Použité technológie

Projekt je vytvorený v jazyku Python. Na spracovanie dát, modelovanie a vizualizáciu sa používajú tieto knižnice:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
openpyxl
```

## Inštalácia knižníc

Pred spustením programu je potrebné nainštalovať potrebné knižnice:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```

## Vstupný súbor

Program očakáva vstupný Excel súbor:

```text
Final Prepared Dataset - Diabetes and Hypertension Data.xlsx
```

Súbor musí byť uložený v rovnakom priečinku ako Python skript. Ak je uložený inde, je potrebné upraviť cestu v riadku:

```python
df = pd.read_excel("Final Prepared Dataset - Diabetes and Hypertension Data.xlsx")
```

## Spustenie programu

Program je možné spustiť v prostredí Visual Studio Code, Jupyter Notebook alebo priamo cez terminál.

Príklad spustenia cez terminál:

```bash
python predictive_analysis.py
```

## Stručný opis programu

1. načítanie vstupného datasetu
2. základný výpis informácií o dátach
3. vytvorenie cieľovej premennej pre modelovanie
4. vytvorenie odvodených premenných
5. rozdelenie vstupných premenných na dve skupiny
6. trénovanie a porovnanie klasifikačných modelov
7. vyhodnotenie modelov pomocou krížovej validácie
8. vyhodnotenie modelov na testovacej množine
9. vykreslenie konfúznych matíc a ROC kriviek
10. optimalizácia modelu XGBoost pomocou Grid Search
11. zobrazenie dôležitosti premenných
12. uloženie výsledkov do suborov

## Použité modely

```text
Logistic Regression
Random Forest
Gradient Boosting
XGBoost
```

Modely sa porovnávajú v dvoch variantoch:

```text
bez behaviorálnych premenných
s behaviorálnymi premennými
```

## Hodnotiace metriky

Výkonnosť modelov sa hodnotí pomocou metrík:

```text
Accuracy
Precision
Recall
F1-score
ROC AUC
```

Okrem číselných metrík program vytvára aj grafické výstupy:

```text
konfúzna matica
ROC krivka
graf dôležitosti premenných
```

## Rozdelenie dát

Pri testovaní sa dáta delia na tréningovú a testovaciu množinu v pomere:

```text
80 % tréningová množina
20 % testovacia množina
```

Rozdelenie je stratifikované, aby bol zachovaný pomer tried v tréningovej aj testovacej množine.

## Grid Search

Pre model XGBoost sa používa optimalizácia hyperparametrov pomocou `GridSearchCV`.

Optimalizované parametre:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
```

Ako hlavná metrika pri optimalizácii sa používa `F1-score`.

## Výstupné súbory

Po dokončení programu sa vytvoria tieto CSV súbory:

```text
model_results_cross_validation.csv
model_results_test_graphs.csv
feature_importance_xgboost_gridsearch.csv
```

### model_results_cross_validation.csv

Obsahuje výsledky modelov získané pomocou 5-násobnej krížovej validácie.

### model_results_test_graphs.csv

Obsahuje výsledky modelov získané na testovacej množine.

### feature_importance_xgboost_gridsearch.csv

Obsahuje dôležitosť jednotlivých premenných vo finálnom modeli XGBoost po optimalizácii hyperparametrov.

## Možné chyby

`ModuleNotFoundError`, niektorá potrebná knižnica nie je nainštalovaná. Chýbajúcu knižnicu je možné doinštalovať cez `pip install`.

Príklad:

```bash
pip install openpyxl
```
*Vysledky su urcene na analyticke a akademicke ucely v ramci bakalarskej prace.
