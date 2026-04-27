# Detección Avanzada de Fraude Financiero con Datos Simulados y Modelos Híbridos

Proyecto final del curso CC3045 – Inteligencia Artificial.

## Integrantes
- José Gerardo Ruiz García - 23719
- Gerardo Andre Fernandez Cruz - 23763
- Humberto Alexander de la Cruz - 23735
- Daniel Oswaldo Juárez Herrera - 23709

## Descripción
Este proyecto busca diseñar un sistema de detección de fraude en transacciones financieras más cercano a producción, replicando el comportamiento transaccional en un entorno con flujo continuo de datos, concept drift y asimetría de costos.

El sistema es híbrido y está compuesto por 3 capas:
1. **Detección Supervisada**: SVM, Decision Trees, Redes Neuronales.
2. **Modelado Secuencial (MDP)**: Detección de patrones de fraude progresivo mediante el estado del cliente a lo largo del tiempo.
3. **Inferencia Probabilística (Redes Bayesianas)**: Estimación de la probabilidad de fraude dado el contexto.

## Estructura del Proyecto
- `data/`: Almacenamiento de datos crudos, procesados y externos.
- `models/`: Modelos entrenados.
- `notebooks/`: Cuadernos Jupyter para experimentación y análisis exploratorio.
- `src/`: Código fuente modularizado.
  - `data/`: Generación de datos sintéticos y preprocesamiento.
  - `models/`: Implementación de modelos supervisados, secuenciales y probabilísticos.
  - `utils/`: Utilidades y métricas.
- `tests/`: Pruebas unitarias.
- `main.py`: Punto de entrada (pipeline) principal.

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución y Uso

El proyecto consta de dos fases principales: el entrenamiento (Backend) y el simulador en vivo (Frontend).

### 1. Entrenar la Inteligencia Artificial (Backend)
Para ejecutar el pipeline completo (generación de datos sintéticos, preprocesamiento con SMOTE, entrenamiento de modelos, ensamblaje Stacking y evaluación):
```bash
python main.py
```
* **¿Qué hace esto?** Genera las transacciones simuladas, entrena las 3 capas de inteligencia artificial y genera gráficos de evaluación (`roc_curves`, matrices de confusión e importancia de variables) dentro de la carpeta `reports/figures/`.
* **Nota:** Este paso exportará los modelos pesados a `.pkl` en la carpeta `models/`, lo cual es obligatorio antes de levantar la interfaz web.

### 2. Simulador de Flujo Continuo (Dashboard Interactivo)
Una vez finalizado el entrenamiento, puedes levantar el simulador en tiempo real usando **Streamlit**:
```bash
streamlit run app.py
```
Esto abrirá automáticamente una pestaña en tu navegador (por defecto en `http://localhost:8501`).

### ¿Cómo interpretar el Simulador Web?
Al hacer clic en **"▶ Iniciar Monitoreo en Vivo"**, el sistema emulará un entorno bancario real:
- **Panel Izquierdo (Streaming):** Muestra transacciones entrando al sistema en tiempo real.
- **Panel Derecho (Motor Híbrido):** 
  - El modelo **Stacking Supervisado** y la **Red Bayesiana** emiten una probabilidad de fraude basada en el historial del cliente y el contexto.
  - El **MDP (Capa de Decisión)** evalúa estas probabilidades considerando el riesgo financiero y decide si *Aprobar (Verde)*, *Rechazar (Rojo)* o *Solicitar 2FA (Amarillo)*.
- **Inteligencia Artificial Explicable (XAI - SHAP):** Si se detecta un posible fraude, el sistema generará dinámicamente un gráfico de cascada (Waterfall) en la esquina inferior derecha, explicando matemáticamente **por qué** la inteligencia artificial tomó esa decisión, aportando transparencia al modelo.
