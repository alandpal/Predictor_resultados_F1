# 🏎️ F1 Championship Predictor

<div align="center">

![F1 Banner](https://img.shields.io/badge/F1-Championship_Predictor-red?style=for-the-badge&logo=f1&logoColor=white)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange?style=for-the-badge)](https://xgboost.ai/)

**Predicción de Clasificación Final del Campeonato de Fórmula 1 usando Machine Learning**

[Características](#-características) • [Instalación](#-instalación) • [Uso](#-uso) • [Resultados](#-resultados)

</div>

---

## 📖 Descripción

Sistema de predicción basado en **Machine Learning** que estima la clasificación final del campeonato de F1 desde cualquier ronda de la temporada actual. Entrenado con datos históricos de 2008-2024.

### 🎯 ¿Qué hace?

Responde preguntas como:
- *"¿Quién será el campeón este año?"*
- *"¿En qué posición terminará cada piloto?"*
- *"¿Puede X alcanzar a Y en el campeonato?"*

**Con 94.6% de precisión** (Spearman Correlation) en la predicción del orden final.

---

## ✨ Características

- **🤖 Sistema Híbrido:** 3 modelos XGBoost especializados por fase (EARLY/MID/LATE)
- **📊 8 Features:** Rendimiento actual + contexto histórico (3 años)
- **🔄 Actualización Automática:** Descarga incremental de datos 2025 desde FastF1
- **📈 Interfaz Interactiva:** App web con Streamlit y gráficos Plotly
- **⚡ Rendimiento:** Predicciones instantáneas con sistema de cache

---

## 🏆 Resultados (Test Set 2023-2024)

| Métrica | Valor |
|---------|-------|
| **Spearman Correlation** | **0.946** |
| **Accuracy Top 3** | **87.0%** |
| **Accuracy Top 10** | **91.1%** |
| **MAE Posiciones** | **1.20** |

### Ejemplo Real: 2024 Ronda 22
- ✅ Top 3 predicho correctamente (VER, NOR, LEC)
- ✅ Top 10 con 90% de aciertos

---

## 🚀 Instalación

### Requisitos
- Python 3.8+
- pip

### Pasos
```bash
# 1. Clonar repositorio
git clone https://github.com/alandpal/Predictor_Resultados_F1.git
cd ML_F1_V5

# 2. Instalar dependencias
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Ejecutar aplicación
cd streamlit_app
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`

---

## 💻 Uso

1. **Iniciar la app** con el comando anterior
2. **Actualizar datos 2025** haciendo clic en "🔄 Actualizar Datos"
3. **Seleccionar ronda** con el slider
4. **Ver predicciones** en tabla y gráfico de evolución

### Interfaz

- **Tabla:** Clasificación final predicha (colores por posición)
- **Gráfico:** Evolución de posiciones por ronda
- **Modelo usado:** Indicado al pie (EARLY/MID/LATE)

---

## 📦 Estructura del Proyecto
```
Predictor_resultados_F1/
├── models/                           # Modelos XGBoost entrenados
│   ├── xgboost_early_v5.pkl
│   ├── xgboost_mid_v5.pkl
│   └── xgboost_late_v5.pkl
│
├── data/processed/                   # Datos procesados y listos
│   ├── f1_features_complete.csv     # Dataset 2008-2024
│   └── context_stats_rolling.pkl    # Stats históricos
│
├── streamlit_app/
│   └── app.py                       # Aplicación web
│
├── notebooks/                        # Análisis y entrenamiento
└── requirements.txt
```

---

## 🛠️ Tecnologías

- **ML:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy, FastF1 API
- **Viz:** Plotly, Streamlit
- **Stats:** SciPy

---

## 🔍 Modelo

### Sistema Híbrido (3 Modelos)

| Fase | Rondas | Características | Spearman |
|------|--------|----------------|----------|
| **EARLY** | R1-R5 | Conservador, alta regularización | 0.789 |
| **MID** | R6-R12 | Balanceado | 0.928 |
| **LATE** | R13+ | Agresivo, alta precisión | **0.981** |

---

## 📊 Validación

- **Train:** 2008-2022 (6,221 registros)
- **Test:** 2023-2024 (919 registros, nunca vistos)
- **CV:** Time Series Split (5 folds)
- **Métrica:** Spearman Correlation (mide orden de clasificación)

---

## ⚙️ Configuración Avanzada

### Actualizar Modelos

Si quieres re-entrenar los modelos con nuevos datos, consulta los notebooks en `/notebooks`:

1. `3_modelado_baseline.ipynb` - Modelo base
2. `4_optimizacion_hiperparametros.ipynb` - Optimización por fase
3. `5_evaluacion_final.ipynb` - Evaluación y métricas

### Datos

Los datos procesados ya están incluidos. Para actualizar:
- Descarga datos históricos con FastF1
- Ejecuta notebooks de procesamiento
- Los modelos se cargan automáticamente

## 👤 Autor

**Albert Andrés**
- GitHub: [@alandpal](https://github.com/alandpal)
- LinkedIn: [Albert Andrés Palop](https://linkedin.com/in/albert-andres-palop)

---

## 🙏 Agradecimientos

- **FastF1:** Por proporcionar la API de datos de F1
- **XGBoost:** Framework de Machine Learning
- **Streamlit:** Framework para la interfaz web

---

<div align="center">

**⭐ Si te gusta este proyecto, dale una estrella en GitHub ⭐**

</div>