# import kniznic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# modelovanie
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
# modely
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
# metriky
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# nacitanie dat
df = pd.read_excel("Final Prepared Dataset - Diabetes and Hypertension Data.xlsx")

print("Rozmer datasetu:", df.shape)

print("\nRozdelenie ADHERENCE:")
print(df["ADHERENCE"].value_counts())

print("\nPercenta ADHERENCE:")
print(df["ADHERENCE"].value_counts(normalize=True))

print("\nPrvych 5 riadkov:")
print(df.head())

# vytvorenie cielovej premenej dropout 
df["Dropout"] = (df["ADHERENCE"] == "NON-ADHERENT").astype(int)

print("\nRozdelenie Dropout:")
print(df["Dropout"].value_counts())

print("\nPercenta Dropout:")
print(df["Dropout"].value_counts(normalize=True))

df = df.drop(columns=["ADHERENCE"])

# vytvorenie novych premennych
df["CLAIM_CONTRIBUTION_RATIO"] = df["ANNUALCLAIMAMOUNT"] / (df["ANNUALCONTRIBUTION"] + 1)

df["COST_PER_UNIT"] = df["ANNUALCLAIMAMOUNT"] / (df["UNITSTOTAL"] + 1)

df["AGE_GROUP"] = pd.cut(
    df["AGE"],
    bins=[0, 30, 45, 60, 75, 120],
    labels=[1, 2, 3, 4, 5]
).astype(int)

df["HIGH_RISK_AGE"] = (df["AGE"] > 65).astype(int)

df["TREATMENT_INTENSITY"] = df["UNITSTOTAL"] / (df["AGE"] + 1)

df["COST_BURDEN"] = df["ANNUALCLAIMAMOUNT"] / (df["AGE"] + 1)

df["COMORBID_RISK"] = (
    df["DIAGNOSIS_HYPERTENSION"].astype(int)
    + (~df["COMORBIDITY_NO_COMORBIDITY"]).astype(int)
)

df["AGE_COMPLICATION_RISK"] = (
    df["HIGH_RISK_AGE"].astype(int)
    + (~df["COMPLICATIONDEVELOPMENT_NO_COMPLICATION"]).astype(int)
)

# rozdelenie premennych na behaviorálne a nebehaviorálne
non_behavioral_cols = [
    "AGE",
    "AGE_GROUP",
    "HIGH_RISK_AGE",
    "GENDER_M",
    "SCHEMETYPE_MEDIUM",
    "SCHEMETYPE_PREMIUM",
    "DIAGNOSIS_HYPERTENSION",
    "COVERTYPE_STANDARD",
    "COMORBIDITY_NO_COMORBIDITY",
    "COMPLICATIONDEVELOPMENT_NO_COMPLICATION",
    "COMORBID_RISK",
    "AGE_COMPLICATION_RISK"
]

behavioral_cols = [
    "ANNUALCONTRIBUTION",
    "ANNUALCLAIMAMOUNT",
    "UNITSTOTAL",
    "CLAIM_CONTRIBUTION_RATIO",
    "COST_PER_UNIT",
    "TREATMENT_INTENSITY",
    "COST_BURDEN"
]

y = df["Dropout"]

X_non_behavioral = df[non_behavioral_cols].copy().astype(float)
X_with_behavioral = df[non_behavioral_cols + behavioral_cols].copy().astype(float)

# funkcia na krizovu validaciu
def evaluate_with_cv(X, y, model, model_name, dataset_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC_AUC": scores["test_roc_auc"].mean()
    }

    print(f"\n===== {model_name} | {dataset_name} =====")
    print("Accuracy :", round(result["Accuracy"], 4))
    print("Precision:", round(result["Precision"], 4))
    print("Recall   :", round(result["Recall"], 4))
    print("F1-score :", round(result["F1"], 4))
    print("ROC AUC  :", round(result["ROC_AUC"], 4))

    return result

# funkcia na grafy a metriky 
def plot_model_evaluation(X, y, model, model_name, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n===== TEST GRAF: {model_name} | {dataset_name} =====")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))
    print("ROC AUC  :", round(auc, 4))
    print("Confusion Matrix:")
    print(cm)

    # konfuzna matica
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens"
    )
    plt.title(f"Confusion Matrix - {model_name} | {dataset_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # roc krivka
    fpr, tpr, _ = roc_curve(y_test, y_proba)
# graf roc krivky
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {model_name} | {dataset_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": auc
    }

# definovanie modelov
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# krizova validacia modelov s a bez behaviorálnych dát
results = []

for model_name, model in models.items():
    results.append(
        evaluate_with_cv(
            X_non_behavioral,
            y,
            model,
            model_name,
            "Bez behaviorálnych dát"
        )
    )

    results.append(
        evaluate_with_cv(
            X_with_behavioral,
            y,
            model,
            model_name,
            "S behaviorálnymi dátami"
        )
    )

results_df = pd.DataFrame(results)

print("\n==============================")
print("PREHLAD VYSLEDKOV - CROSS VALIDATION")
print("==============================")
print(results_df.sort_values(by="ROC_AUC", ascending=False))

# Grafy pre modely bez Grid Search
test_results = []

for model_name, model in models.items():

    test_results.append(
        plot_model_evaluation(
            X_non_behavioral,
            y,
            model,
            model_name,
            "Bez behaviorálnych dát"
        )
    )

    test_results.append(
        plot_model_evaluation(
            X_with_behavioral,
            y,
            model,
            model_name,
            "S behaviorálnymi dátami"
        )
    )

test_results_df = pd.DataFrame(test_results)

print("\n==============================")
print("PREHLAD TEST VYSLEDKOV - BEZ GRID SEARCH")
print("==============================")
print(test_results_df.sort_values(by="ROC_AUC", ascending=False))

# XGBoost bez Grid Search
xgb_normal = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

plot_model_evaluation(
    X_with_behavioral,
    y,
    xgb_normal,
    "XGBoost bez Grid Search",
    "S behaviorálnymi dátami"
)

plot_model_evaluation(
    X_non_behavioral,
    y,
    xgb_normal,
    "XGBoost bez Grid Search",
    "Bez behaviorálnych dát"
)

# grid search pre xgboost
print("\n==============================")
print("GRID SEARCH PRE XGBOOST")
print("==============================")

X_train_beh, X_test_beh, y_train_beh, y_test_beh = train_test_split(
    X_with_behavioral,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# základný model pre grid search
xgb_base = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)
# definovanie rozsahu hyperparametrov pre grid search
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid search
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_beh, y_train_beh)

print("\nNajlepsie parametre pre XGBoost:")
print(grid_search.best_params_)

print("\nNajlepsie F1 z Grid Search:")
print(round(grid_search.best_score_, 4))

# finalny xgboost po grid search
final_model = grid_search.best_estimator_

final_model.fit(X_train_beh, y_train_beh)

y_pred = final_model.predict(X_test_beh)
y_proba = final_model.predict_proba(X_test_beh)[:, 1]

final_acc = accuracy_score(y_test_beh, y_pred)
final_prec = precision_score(y_test_beh, y_pred)
final_rec = recall_score(y_test_beh, y_pred)
final_f1 = f1_score(y_test_beh, y_pred)
final_auc = roc_auc_score(y_test_beh, y_proba)
final_cm = confusion_matrix(y_test_beh, y_pred)

print("\n==============================")
print("FINALNY XGBOOST PO GRID SEARCH - TEST SET")
print("==============================")
print("Accuracy :", round(final_acc, 4))
print("Precision:", round(final_prec, 4))
print("Recall   :", round(final_rec, 4))
print("F1-score :", round(final_f1, 4))
print("ROC AUC  :", round(final_auc, 4))
print("Confusion Matrix:")
print(final_cm)

# konfuzna matica finalneho modelu
plt.figure(figsize=(6, 5))
sns.heatmap(
    final_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)
plt.title("Confusion Matrix - XGBoost Grid Search | S behaviorálnymi dátami")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ROC krivka finalneho modelu
fpr, tpr, _ = roc_curve(y_test_beh, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {final_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - XGBoost Grid Search | S behaviorálnymi dátami")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# dolezitost jednotlivych premennych pre finalny model
feature_importance = pd.DataFrame({
    "Feature": X_with_behavioral.columns,
    "Importance": final_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature importance:")
print(feature_importance)
# graf dolezitosti premennych
plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance - XGBoost Grid Search")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# vysledky ulozene do csv suborov
results_df.to_csv("model_results_cross_validation.csv", index=False)
test_results_df.to_csv("model_results_test_graphs.csv", index=False)
feature_importance.to_csv("feature_importance_xgboost_gridsearch.csv", index=False)