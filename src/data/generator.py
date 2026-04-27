import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TransactionSimulator:
    def __init__(self, num_customers=1000, num_merchants=100, fraud_ratio=0.02):
        self.num_customers = num_customers
        self.num_merchants = num_merchants
        self.fraud_ratio = fraud_ratio

    def generate_data(self, num_transactions=10000) -> pd.DataFrame:
        """
        Genera un dataset sintético de transacciones financieras.
        """
        print(f"Generating {num_transactions} synthetic transactions...")
        
        np.random.seed(42)
        
        customer_ids = [f"CUST_{i}" for i in range(self.num_customers)]
        merchant_ids = [f"MERCH_{i}" for i in range(self.num_merchants)]
        
        # Generar fechas en un rango de 30 días
        start_date = datetime.now() - timedelta(days=30)
        
        data = []
        for _ in range(num_transactions):
            is_fraud = np.random.choice([0, 1], p=[1 - self.fraud_ratio, self.fraud_ratio])
            
            # Comportamiento anómalo si es fraude
            if is_fraud:
                amount = np.random.uniform(500, 5000)
                location_type = np.random.choice(['INTERNATIONAL', 'HIGH_RISK'])
            else:
                amount = np.random.exponential(50) + 5  # Montos comunes
                location_type = np.random.choice(['LOCAL', 'NATIONAL'])
            
            # Fecha aleatoria
            delta = timedelta(days=np.random.randint(0, 30), minutes=np.random.randint(0, 1440))
            txn_date = start_date + delta
            
            data.append({
                'customer_id': np.random.choice(customer_ids),
                'merchant_id': np.random.choice(merchant_ids),
                'amount': round(amount, 2),
                'location_type': location_type,
                'timestamp': txn_date,
                'is_fraud': is_fraud
            })
            
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
