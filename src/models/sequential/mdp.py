import numpy as np
import pandas as pd

class FraudMDP:
    """
    Markov Decision Process para detección de fraude.
    Utiliza Value Iteration para encontrar la política óptima.
    """
    def __init__(self, gamma=0.9, theta=1e-4):
        self.gamma = gamma # Factor de descuento
        self.theta = theta # Umbral de convergencia
        
        self.states = ['LOW_RISK', 'MEDIUM_RISK', 'HIGH_RISK']
        self.actions = ['APPROVE', 'DECLINE', 'REQUIRE_2FA']
        
        # P(s' | s, a) - Matriz de transición empírica
        self.transitions = {s: {a: {} for a in self.actions} for s in self.states}
        
        # R(s, a) - Recompensas esperadas
        self.rewards = {s: {a: 0.0 for a in self.actions} for s in self.states}
        
        # V(s) - Valores de los estados
        self.V = {s: 0.0 for s in self.states}
        
        # Política óptima: pi(s) -> a
        self.policy = {s: 'APPROVE' for s in self.states}

    def _assign_state(self, row):
        if row['amount'] > 3000 or row.get('txn_count_last_1d', 0) > 5:
            return 'HIGH_RISK'
        elif row['amount'] > 1000 or row.get('txn_count_last_1d', 0) > 3:
            return 'MEDIUM_RISK'
        return 'LOW_RISK'

    def fit(self, transactions: pd.DataFrame):
        """
        Aprende P(s'|s, a) y R(s, a) de los datos históricos y ejecuta Value Iteration.
        """
        print("Training MDP (Estimating empirical transitions and rewards)...")
        df = transactions.copy()
        
        # Asignar estados empíricos
        df['state'] = df.apply(self._assign_state, axis=1)
        
        # Matriz de costos asimétricos del mundo real
        # Costos y beneficios relativos para modelar la decisión de negocio
        cost_matrix = {
            'APPROVE': {'legit': 1, 'fraud': -100},      # Mucho castigo si aprobamos fraude
            'DECLINE': {'legit': -10, 'fraud': 20},      # Castigo por fricción, premio por parar fraude
            'REQUIRE_2FA': {'legit': -2, 'fraud': 5}     # Fricción moderada, filtro moderado
        }
        
        for s in self.states:
            subset = df[df['state'] == s]
            if len(subset) == 0: 
                continue
            
            p_fraud = subset['is_fraud'].mean()
            p_legit = 1 - p_fraud
            
            for a in self.actions:
                # Recompensa esperada
                self.rewards[s][a] = (p_legit * cost_matrix[a]['legit']) + (p_fraud * cost_matrix[a]['fraud'])
                
                # Transiciones (Heurística basada en la acción y probabilidad subyacente)
                if a == 'APPROVE':
                    self.transitions[s][a]['HIGH_RISK'] = p_fraud
                    self.transitions[s][a][s] = p_legit
                elif a == 'DECLINE':
                    self.transitions[s][a]['LOW_RISK'] = 1.0 # Resetea el riesgo cortando la operación
                elif a == 'REQUIRE_2FA':
                    self.transitions[s][a]['LOW_RISK'] = 0.8
                    self.transitions[s][a][s] = 0.2

        # Normalizar probabilidades
        for s in self.states:
            for a in self.actions:
                total = sum(self.transitions[s][a].values())
                if total > 0:
                    for s_prime in self.transitions[s][a]:
                        self.transitions[s][a][s_prime] /= total

        self._value_iteration()

    def _value_iteration(self):
        """
        Algoritmo de Value Iteration para resolver el MDP.
        """
        print("Running Value Iteration...")
        iterations = 0
        while True:
            delta = 0
            new_V = self.V.copy()
            for s in self.states:
                action_values = []
                for a in self.actions:
                    val = self.rewards[s][a]
                    for s_prime in self.transitions[s][a]:
                        val += self.gamma * self.transitions[s][a][s_prime] * self.V[s_prime]
                    action_values.append((a, val))
                
                best_action, best_val = max(action_values, key=lambda x: x[1])
                new_V[s] = best_val
                self.policy[s] = best_action
                delta = max(delta, abs(self.V[s] - new_V[s]))
                
            self.V = new_V
            iterations += 1
            if delta < self.theta:
                break
        print(f"Value Iteration converged in {iterations} iterations.")
        print(f"Optimal Policy found: {self.policy}")

    def decide(self, features: dict) -> str:
        """
        Devuelve la acción óptima según la política calculada.
        """
        row = {'amount': features.get('amount', 0), 'txn_count_last_1d': features.get('txn_count_last_1d', 0)}
        current_state = self._assign_state(row)
        return self.policy.get(current_state, 'APPROVE')
