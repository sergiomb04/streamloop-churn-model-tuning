"""
StreamLoop — Churn Model Tuning
Entrenamiento, optimización y evaluación de la línea base vs modelo ajustado.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def load_and_clean_data(url: str):
    print("-> Cargando dataset de clientes...")
    df = pd.read_csv(url)
    
    # Limpieza mínima
    df = df.drop(columns=['customerID'])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y


def build_preprocessor(X: pd.DataFrame):
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return preprocessor


def evaluate_model(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'False Negatives (FN)': int(cm[1, 0]),
        'False Positives (FP)': int(cm[0, 1]),
    }
    return metrics


def main():
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    X, y = load_and_clean_data(url)

    # 1. División Train/Test antes de cualquier transformación
    print("-> Dividiendo en Train (80%) y Test (20%) estratificado...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # 2. Línea Base con hiperparámetros por defecto
    print("\n[1/3] Entrenando Línea Base (RandomForest con defaults)...")
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_pipeline, X_test, y_test, "Línea Base")

    # 3. Búsqueda Amplia con RandomizedSearchCV
    print("\n[2/3] Ejecutando búsqueda amplia con RandomizedSearchCV (scoring='recall')...")
    param_dist = {
        'classifier__n_estimators': [50, 100, 150, 200, 250],
        'classifier__max_depth': [3, 4, 6, 8, 12, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None],
        'classifier__max_features': ['sqrt', 'log2', None]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
        param_distributions=param_dist,
        n_iter=25,
        scoring='recall',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True
    )
    rs.fit(X_train, y_train)
    print(f"   Mejor Recall en CV (RandomizedSearchCV): {rs.best_score_:.4f}")

    # 4. Refinamiento Focalizado con GridSearchCV
    print("\n[3/3] Refinando espacio focalizado con GridSearchCV...")
    param_grid = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__max_depth': [4, 5, 6],
        'classifier__min_samples_leaf': [2, 4],
        'classifier__min_samples_split': [5, 10],
        'classifier__class_weight': ['balanced', 'balanced_subsample'],
        'classifier__max_features': ['log2', 'sqrt']
    }
    gs = GridSearchCV(
        estimator=Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
        param_grid=param_grid,
        scoring='recall',
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    gs.fit(X_train, y_train)
    print(f"   Mejor Recall en CV (GridSearchCV): {gs.best_score_:.4f}")
    print(f"   Mejores hiperparámetros: {gs.best_params_}")

    # 5. Evaluación Final en Test Set
    tuned_metrics = evaluate_model(gs.best_estimator_, X_test, y_test, "Modelo Ajustado")

    # Tabla Comparativa
    df_compare = pd.DataFrame({
        'Línea Base': baseline_metrics,
        'Modelo Ajustado': tuned_metrics
    })
    df_compare['Diferencia'] = df_compare['Modelo Ajustado'] - df_compare['Línea Base']
    
    print("\n" + "=" * 55)
    print("  RESULTADOS COMPARATIVOS FINALES (TEST SET)")
    print("=" * 55)
    print(df_compare.to_string())
    print("=" * 55)
    print(f"Reducción de cancelaciones no detectadas: {baseline_metrics['False Negatives (FN)']} -> {tuned_metrics['False Negatives (FN)']}")


if __name__ == "__main__":
    main()