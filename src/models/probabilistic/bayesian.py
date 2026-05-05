from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

class FraudBayesianNetwork:
    """
    Red Bayesiana para la inferencia probabilística del riesgo de fraude.
    Estructura ampliada para mayor realismo y uso de BayesianEstimator.
    """
    def __init__(self):
        # Definición del Grafo Acíclico Dirigido (DAG)
        self.model = DiscreteBayesianNetwork([
            ('location_type', 'is_fraud'),
            ('hour_bin', 'is_fraud'),
            ('amount_bin', 'is_fraud'),
            ('is_fraud', 'alert_triggered')
        ])
        
    def fit(self, data: pd.DataFrame):
        """
        Aprende las Tablas de Probabilidad Condicional (CPDs) usando BayesianEstimator.
        """
        print("Training Bayesian Network (Estimating CPDs)...")
        df_bayes = data.copy()
        
        # Discretización avanzada requerida para PGM
        if 'amount' in df_bayes.columns:
            df_bayes['amount_bin'] = pd.qcut(df_bayes['amount'], q=4, labels=['LOW', 'MED', 'HIGH', 'VERY_HIGH'])
            
        if 'timestamp' in df_bayes.columns:
            df_bayes['hour'] = df_bayes['timestamp'].dt.hour
            df_bayes['hour_bin'] = pd.cut(df_bayes['hour'], bins=[-1, 6, 12, 18, 24], labels=['NIGHT', 'MORNING', 'AFTERNOON', 'EVENING'])
            
        # Generar nodo condicionado hijo (efecto)
        if 'alert_triggered' not in df_bayes.columns:
            df_bayes['alert_triggered'] = (df_bayes['is_fraud'] == 1) | (df_bayes['amount_bin'].isin(['HIGH', 'VERY_HIGH']))
            df_bayes['alert_triggered'] = df_bayes['alert_triggered'].astype(str)
            
        # Convertir variables discretas a strings explícitamente (evita errores en pgmpy)
        df_bayes['is_fraud'] = df_bayes['is_fraud'].astype(str)
        df_bayes['location_type'] = df_bayes['location_type'].astype(str)
        df_bayes['amount_bin'] = df_bayes['amount_bin'].astype(str)
        df_bayes['hour_bin'] = df_bayes['hour_bin'].astype(str)
        
        # Estimar CPDs usando BDeu score (mucho más robusto que Maximum Likelihood en casos de datos escasos)
        estimator = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=10)
        self.model.fit(df_bayes[['location_type', 'hour_bin', 'amount_bin', 'is_fraud', 'alert_triggered']], 
                       estimator=estimator)
        
        # Validar consistencia del modelo
        self.model.check_model()
        print("Bayesian Network learned successfully.")
        
        # Mostrar una CPD como ejemplo para la documentación del proyecto
        print("\nEjemplo de Tabla de Probabilidad Condicional (CPD) aprendida para 'alert_triggered':")
        print(self.model.get_cpds('alert_triggered'))
                       
    def predict_proba(self, evidence: dict) -> float:
        """
        Inferencia exacta mediante Variable Elimination.
        Ejemplo evidence: {'amount_bin': 'VERY_HIGH', 'hour_bin': 'NIGHT'}
        """
        infer = VariableElimination(self.model)
        
        # Asegurarnos de que la evidencia sea string para coincidir con las tablas de pgmpy
        evidence_str = {k: str(v) for k, v in evidence.items()}
        
        result = infer.query(variables=['is_fraud'], evidence=evidence_str)
        
        # Obtener la probabilidad donde is_fraud = '1'
        state_names = result.state_names['is_fraud']
        if '1' in state_names:
            fraud_idx = state_names.index('1')
            return result.values[fraud_idx]
        return 0.0
