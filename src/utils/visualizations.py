import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import pandas as pd
import os
import shap

def plot_confusion_matrix(y_true, y_pred, model_name):
    os.makedirs('reports/figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_{model_name}.png')
    plt.close()

def plot_roc_curve(model, X_test, y_test, model_name):
    os.makedirs('reports/figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.title(f'Curva ROC - {model_name}')
    # plt.plot([0, 1], [0, 1], 'k--', label='Azar') # Línea de azar
    plt.tight_layout()
    plt.savefig(f'reports/figures/roc_curve_{model_name}.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    # Solo árboles de decisión tienen feature_importances_
    if not hasattr(model, 'feature_importances_'):
        return
    
    os.makedirs('reports/figures', exist_ok=True)
    importances = model.feature_importances_
    df = pd.DataFrame({'Variable': feature_names, 'Importancia': importances})
    df = df.sort_values('Importancia', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x='Importancia', y='Variable', ax=ax, palette='viridis', hue='Variable', legend=False)
    plt.title(f'Importancia de Variables - {model_name}')
    plt.tight_layout()
    plt.savefig(f'reports/figures/feature_importance_{model_name}.png')
    plt.close()

def plot_shap_summary(model, X_test, model_name):
    """
    Genera el gráfico SHAP para XAI (Inteligencia Artificial Explicable).
    """
    try:
        os.makedirs('reports/figures', exist_ok=True)
        # Usar TreeExplainer para modelos basados en árboles, o KernelExplainer/Explainer genérico
        # Por rendimiento, usamos un muestreo de X_test
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        
        # Para DecisionTree o Stacking con clasificador final interpretable
        if hasattr(model, 'estimators_') or model_name == 'DecisionTree':
            # Intentar TreeExplainer si es árbol
            if model_name == 'DecisionTree':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                # Si shap_values es una lista (clasificación multiclase), tomar clase 1
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample).values
                
            plt.figure(figsize=(8, 5))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f'SHAP XAI Summary - {model_name}')
            plt.tight_layout()
            plt.savefig(f'reports/figures/shap_summary_{model_name}.png')
            plt.close()
    except Exception as e:
        print(f"Aviso: No se pudo generar gráfico SHAP para {model_name} ({e})")

