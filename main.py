import os
import warnings

# Suprimir warnings visuales (ConvergenceWarning de Sklearn y FutureWarning de pgmpy)
warnings.filterwarnings("ignore")

from src.data.generator import TransactionSimulator
from src.data.preprocessor import DataPreprocessor
from src.models.supervised.classifiers import SupervisedModels
from src.models.sequential.mdp import FraudMDP
from src.models.probabilistic.bayesian import FraudBayesianNetwork

from sklearn.model_selection import train_test_split
import joblib

def main():
    print("=====================================================")
    print(" Detección Avanzada de Fraude Financiero")
    print(" Modelos Híbridos (Supervisado, MDP, Bayesiano)")
    print("=====================================================\n")
    
    # 1. Generación de Datos Simulados
    print("--- Fase 1: Generación de Datos ---")
    simulator = TransactionSimulator(num_customers=500, num_merchants=50, fraud_ratio=0.05)
    df_raw = simulator.generate_data(num_transactions=3000)
    
    # Añadir ruido a los datos para simular imperfecciones del mundo real (Concept Drift / Errores)
    # Esto reduce el Accuracy/ROC a valores realistas (< 1.0) para el análisis académico
    noise_idx = df_raw.sample(frac=0.1, random_state=42).index
    df_raw.loc[noise_idx, 'is_fraud'] = 1 - df_raw.loc[noise_idx, 'is_fraud']
    
    os.makedirs("data/raw", exist_ok=True)
    df_raw.to_csv("data/raw/transactions.csv", index=False)
    print(f"Datos crudos generados y guardados: {df_raw.shape[0]} transacciones.\n")
    
    # 2. Preprocesamiento y Feature Engineering
    print("--- Fase 2: Preprocesamiento y Balanceo ---")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df_raw)
    
    # Train-test split (Estratificado)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Balanceo de Clases usando SMOTE
    X_train_bal, y_train_bal = preprocessor.balance_data(X_train, y_train)
    
    # GUARDAR PREPROCESADOR PARA EL DASHBOARD
    preprocessor.save()
    print("")

    # 3. Modelos Supervisados (Capa 1)
    print("--- Capa 1: Modelos Supervisados (con GridSearch y Ensamblaje Meta-Algorítmico) ---")
    supervised = SupervisedModels()
    # Entrenar modelos base y el Stacking (Ensamblaje)
    for model_name in ['DecisionTree', 'NeuralNetwork', 'SVM', 'Stacking']:
        # Optimizamos los rápidos, SVM y Stacking pueden ser lentos en GridSearch
        optimize = (model_name in ['DecisionTree', 'NeuralNetwork']) 
        supervised.train(model_name, X_train_bal, y_train_bal, optimize=optimize)
        supervised.evaluate(model_name, X_test, y_test)
        
    # 4. Modelado Secuencial (Capa 2)
    print("\n--- Capa 2: Procesos de Decisión de Markov (MDP) ---")
    mdp = FraudMDP(gamma=0.9)
    mdp.fit(df_raw)
    joblib.dump(mdp, "models/mdp.pkl") # Guardar para dashboard
    
    test_feature = {'amount': 3500, 'txn_count_last_1d': 6}
    decision = mdp.decide(test_feature)
    print(f"\nDecisión MDP evaluada para características de ALTO RIESGO {test_feature}:")
    print(f"Acción óptima a tomar: {decision}\n")
    
    # 5. Inferencia Probabilística (Capa 3)
    print("--- Capa 3: Redes Bayesianas ---")
    bn = FraudBayesianNetwork()
    bn.fit(df_raw)
    joblib.dump(bn, "models/bayesian.pkl") # Guardar para dashboard
    
    try:
        # Consultas de inferencia contextuales (demuestra la flexibilidad del modelo Bayesiano)
        prob_night = bn.predict_proba({'amount_bin': 'VERY_HIGH', 'hour_bin': 'NIGHT'})
        prob_morning = bn.predict_proba({'amount_bin': 'LOW', 'hour_bin': 'MORNING'})
        
        print(f"\nInferencia Bayesiana Contextual:")
        print(f"P(Fraude | Monto=MUY ALTO, Hora=NOCHE) = {prob_night:.4f}")
        print(f"P(Fraude | Monto=BAJO, Hora=MAÑANA) = {prob_morning:.4f}\n")
    except Exception as e:
        print(f"Error en inferencia bayesiana: {e}\n")

    print("\n=====================================================")
    print(" Pipeline completado exitosamente.")
    print(" Las gráficas de resultados se han guardado en la carpeta: 'reports/figures/'")
    print("=====================================================\n")

if __name__ == "__main__":
    main()
