from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import joblib
import os
from src.utils.visualizations import plot_confusion_matrix, plot_roc_curve, plot_feature_importance, plot_shap_summary

class SupervisedModels:
    def __init__(self):
        # Modelos base
        base_models = [
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=10)),
            ('nn', MLPClassifier(solver='sgd', max_iter=300, random_state=42, hidden_layer_sizes=(32,))),
        ]
        # SVM es muy lento para stacking por defecto, usamos uno rápido
        
        self.models = {
            'SVM': SVC(probability=True, random_state=42),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'NeuralNetwork': MLPClassifier(solver='sgd', max_iter=500, random_state=42),
            'Stacking': StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(),
                cv=3,
                n_jobs=-1
            )
        }
        
        # Grillas de hiperparámetros para búsqueda
        self.param_grids = {
            'DecisionTree': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'NeuralNetwork': {
                'hidden_layer_sizes': [(32,), (64, 32)],
                'alpha': [0.0001, 0.001]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
        
        self.trained_models = {}
        os.makedirs("models", exist_ok=True)
        
    def train(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series, optimize: bool = False):
        print(f"Training {model_name}...")
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not supported.")
            
        if optimize and model_name in self.param_grids:
            print(f"Performing Hyperparameter tuning (GridSearchCV) for {model_name}...")
            # GridSearchCV evaluará múltiples combinaciones usando validación cruzada
            grid_search = GridSearchCV(model, self.param_grids[model_name], cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best params for {model_name}: {grid_search.best_params_}")
            self.trained_models[model_name] = best_model
        else:
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            
        # Guardar modelo entrenado para futura inferencia sin reentrenar
        joblib.dump(self.trained_models[model_name], f"models/{model_name}.pkl")

        
    def evaluate(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series):
        model = self.trained_models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} has not been trained yet.")
            
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        print(f"\n--- Evaluation: {model_name} ---")
        print(f"Generando y guardando gráficas para {model_name}...")
        
        # Guardar gráficas
        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_roc_curve(model, X_test, y_test, model_name)
        plot_feature_importance(model, X_test.columns, model_name)
        
        # XAI: SHAP solo para algunos modelos para no demorar demasiado
        if model_name in ['DecisionTree', 'Stacking']:
            print(f"Generando explicabilidad XAI (SHAP) para {model_name}...")
            plot_shap_summary(model, X_test, model_name)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}\n")
        return {"roc_auc": roc_auc}
