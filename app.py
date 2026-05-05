import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import shap
import matplotlib.pyplot as plt
import os
from src.data.preprocessor import DataPreprocessor

# Configuración de página
st.set_page_config(page_title="Fraud Guard - AI", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# CSS Avanzado para mejorar la estética
st.markdown("""
    <style>
    .metric-card { 
        background-color: #1E2127; padding: 20px; border-radius: 12px; 
        text-align: center; border: 1px solid #333; box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    .risk-high { color: #ff4b4b; font-weight: 900; font-size: 28px; text-transform: uppercase;}
    .risk-medium { color: #faca2b; font-weight: 900; font-size: 28px; text-transform: uppercase;}
    .risk-low { color: #00cc96; font-weight: 900; font-size: 28px; text-transform: uppercase;}
    .title-text { color: #a3a8b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        preprocessor = DataPreprocessor.load()
        stacking_model = joblib.load("models/Stacking.pkl")
        mdp = joblib.load("models/mdp.pkl")
        bayes = joblib.load("models/bayesian.pkl")
        return preprocessor, stacking_model, mdp, bayes
    except Exception as e:
        return None, None, None, None

preprocessor, stacking_model, mdp, bayes = load_models()

# SIDEBAR
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>FraudGuard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Control de Simulación")
    start_sim = st.button("▶ Iniciar Monitoreo en Vivo", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("### Motor Híbrido Activo")
    st.success("Capa 1: Stacking Classifier")
    st.success("Capa 2: Markov Decision Process")
    st.success("Capa 3: Red Bayesiana")
    
if not all([preprocessor, stacking_model, mdp, bayes]):
    st.error("Modelos no encontrados. Por favor corre `python main.py` en tu terminal primero.")
    st.stop()

# MAIN DASHBOARD
st.title("Centro de Mando Anti-Fraude (Streaming)")
st.markdown("Sistema de detección híbrido evaluando transacciones entrantes con Inteligencia Artificial Explicable.")

# Contenedores vacíos para actualizar dinámicamente
kpi_container = st.empty()
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Últimas Transacciones (Flujo en Vivo)")
    placeholder_table = st.empty()
    st.subheader("Explicabilidad IA (SHAP)")
    placeholder_xai = st.empty()

with col2:
    st.subheader("Decisión del Motor en Tiempo Real")
    placeholder_decision = st.empty()

df_raw = pd.read_csv("data/raw/transactions.csv")
# Streaming más largo y aleatorio
streaming_data = df_raw.sample(100, random_state=np.random.randint(0, 1000)).reset_index(drop=True)

if start_sim:
    history = []
    decisions_xai = {'APPROVE': None, 'REQUIRE_2FA': None, 'DECLINE': None}
    analyzed_count = 0
    blocked_count = 0
    
    for idx, row in streaming_data.iterrows():
        analyzed_count += 1
        history.insert(0, row.to_dict())
        if len(history) > 8:
            history.pop()
            
        display_df = pd.DataFrame(history)
        # Mostrar tabla sin etiqueta real
        placeholder_table.dataframe(display_df.drop(columns=['is_fraud'], errors='ignore'), use_container_width=True, hide_index=True)
        
        # --- PREPROCESAMIENTO ---
        row_df = pd.DataFrame([row])
        row_features = preprocessor.feature_engineering(row_df)
        
        try:
            for col in ['customer_id', 'merchant_id', 'location_type']:
                if row_features[col].iloc[0] in preprocessor.label_encoders[col].classes_:
                    row_features[col] = preprocessor.label_encoders[col].transform(row_features[col])
                else:
                    row_features[col] = 0 
        except Exception:
            pass
            
        features = ['customer_id', 'merchant_id', 'amount', 'location_type', 'hour', 'day_of_week', 'txn_count_last_1d', 'avg_amount']
        X = row_features[features]
        X_scaled = pd.DataFrame(preprocessor.scaler.transform(X), columns=features)
        
        # --- PREDICCIONES ---
        prob_fraud = stacking_model.predict_proba(X_scaled)[0][1]
        mdp_action = mdp.decide({'amount': row['amount'], 'txn_count_last_1d': row_features['txn_count_last_1d'].iloc[0]})
        
        amount_bin = 'LOW'
        if row['amount'] > 2500: amount_bin = 'VERY_HIGH'
        elif row['amount'] > 1000: amount_bin = 'HIGH'
        elif row['amount'] > 500: amount_bin = 'MED'
            
        hour = row_features['hour'].iloc[0]
        hour_bin = 'MORNING'
        if hour < 6: hour_bin = 'NIGHT'
        elif hour > 18: hour_bin = 'EVENING'
        elif hour > 12: hour_bin = 'AFTERNOON'
            
        bayes_prob = bayes.predict_proba({'amount_bin': amount_bin, 'hour_bin': hour_bin})
        
        if mdp_action == 'DECLINE':
            blocked_count += 1
            
        # --- ACTUALIZAR KPIS ---
        with kpi_container.container():
            k1, k2, k3 = st.columns(3)
            k1.metric("Transacciones Analizadas", analyzed_count)
            k2.metric("Fraudes Bloqueados", blocked_count)
            k3.metric("Tasa de Bloqueo", f"{(blocked_count/max(1, analyzed_count))*100:.1f}%")
            st.markdown("<br>", unsafe_allow_html=True)
            
        # --- ACTUALIZAR UI DE DECISIÓN ---
        with placeholder_decision.container():
            risk_color = "risk-low"
            icon = "✅"
            if mdp_action == 'DECLINE': 
                risk_color = "risk-high"
                icon = "🚨"
            elif mdp_action == 'REQUIRE_2FA': 
                risk_color = "risk-medium"
                icon = "📱"
                
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='title-text'>Acción Emitida (MDP)</div>
                    <div class='{risk_color}'>{icon} {mdp_action.replace('_', ' ')}</div>
                    <hr style='border-color: #333;'>
                    <div style='text-align: left;'>
                        <p><b>Cliente:</b> {row['customer_id']}</p>
                        <p><b>Monto:</b> ${row['amount']:.2f}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='metric-card'><div class='title-text'>Stacking ML</div><h3 style='margin:0;'>{prob_fraud*100:.1f}%</h3></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><div class='title-text'>Red Bayesiana</div><h3 style='margin:0;'>{bayes_prob*100:.1f}%</h3></div>", unsafe_allow_html=True)
            
        # --- ACTUALIZAR XAI ---
        decisions_xai[mdp_action] = {
            'X_scaled': X_scaled.copy(),
            'features': features,
            'prob_fraud': prob_fraud,
            'amount': row['amount']
        }
        
        with placeholder_xai.container():
            tabs = st.tabs(["✅ Aprobadas", "📱 Solicita 2FA", "🚨 Rechazadas"])
            tab_mapping = {'APPROVE': tabs[0], 'REQUIRE_2FA': tabs[1], 'DECLINE': tabs[2]}
            
            for action_key, tab in tab_mapping.items():
                with tab:
                    data = decisions_xai[action_key]
                    if data is None:
                        st.info(f"Aún no hay transacciones para la decisión: {action_key.replace('_', ' ')}")
                    else:
                        st.write(f"**Último Monto:** ${data['amount']:.2f} | **Prob. Fraude (Stacking):** {data['prob_fraud']*100:.1f}%")
                        
                        if action_key == 'APPROVE' and data['prob_fraud'] <= 0.4:
                            st.success("Transacción de bajo riesgo. Comportamiento normal. No requiere justificación gráfica.")
                        else:
                            try:
                                dt_model = joblib.load("models/DecisionTree.pkl")
                                explainer = shap.TreeExplainer(dt_model)
                                shap_values = explainer.shap_values(data['X_scaled'])
                                
                                fig, ax = plt.subplots(figsize=(6, 3))
                                # Estilo oscuro para matplotlib
                                plt.style.use('dark_background')
                                ax.set_facecolor('#0E1117')
                                fig.patch.set_facecolor('#0E1117')
                                
                                shap.waterfall_plot(shap.Explanation(values=shap_values[1][0], 
                                                                     base_values=explainer.expected_value[1], 
                                                                     data=data['X_scaled'].iloc[0], 
                                                                     feature_names=data['features']), show=False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            except Exception:
                                if action_key == 'APPROVE':
                                    st.success("✅ **Resolución:** Transacción aprobada. Comportamiento dentro de los parámetros normales para este cliente.")
                                elif action_key == 'REQUIRE_2FA':
                                    st.warning("📱 **Resolución:** Se han detectado patrones inusuales o un comportamiento anómalo moderado. Se recomienda un desafío de autenticación en dos pasos (2FA).")
                                elif action_key == 'DECLINE':
                                    st.error("🚨 **Resolución:** Transacción de alto riesgo. Múltiples factores indican alta probabilidad de fraude. La operación ha sido bloqueada preventivamente.")
            
        time.sleep(1.2) # Ritmo de simulación
