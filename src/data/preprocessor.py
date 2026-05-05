import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from typing import Tuple
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea nuevas variables basadas en el tiempo y comportamiento.
        """
        print("Performing Feature Engineering...")
        df = df.copy()
        
        # Asegurar que timestamp sea datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Variables de tiempo
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Frecuencia de transacción por usuario (rolling count simplificado)
        df['txn_count_last_1d'] = df.groupby('customer_id')['timestamp'].transform(
            lambda x: x.diff().dt.days <= 1).astype(int)
            
        # Monto promedio por usuario
        df['avg_amount'] = df.groupby('customer_id')['amount'].transform('mean')
            
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Codifica variables categóricas, escala numéricas y separa (X, y).
        """
        df = self.feature_engineering(df)
        
        # Label Encoding
        for col in ['customer_id', 'merchant_id', 'location_type']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        features = ['customer_id', 'merchant_id', 'amount', 'location_type', 'hour', 'day_of_week', 'txn_count_last_1d', 'avg_amount']
        X = df[features]
        y = df['is_fraud']
        
        # Scaling
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=features)
        
        return X_scaled, y

    def balance_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balancea el dataset usando SMOTE (Synthetic Minority Over-sampling Technique).
        """
        print(f"Balancing data with SMOTE. Original shape: {X.shape}, Frauds: {y.sum()}")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"Resampled shape: {X_resampled.shape}, Frauds: {y_resampled.sum()}")
        return X_resampled, y_resampled

    def save(self, filepath: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str = "models/preprocessor.pkl"):
        return joblib.load(filepath)

