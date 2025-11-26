# ============================================================================
# APP STREAMLIT - F1 CHAMPIONSHIP PREDICTOR
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path
import fastf1
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURACIÓN ====================

st.set_page_config(
    page_title="F1 Championship Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODELS_DIR = Path('models')
DATA_DIR = Path('data/processed')
CACHE_2025_FILE = DATA_DIR / 'f1_2025_local.csv'
CACHE_INFO_FILE = DATA_DIR / 'f1_2025_info.json'

# ==================== CARGAR MODELOS Y CONFIG ====================

@st.cache_resource
def cargar_modelos():
    """Carga los 3 modelos y configuración"""
    
    with open(MODELS_DIR / 'xgboost_early_v5.pkl', 'rb') as f:
        modelo_early = pickle.load(f)
    
    with open(MODELS_DIR / 'xgboost_mid_v5.pkl', 'rb') as f:
        modelo_mid = pickle.load(f)
    
    with open(MODELS_DIR / 'xgboost_late_v5.pkl', 'rb') as f:
        modelo_late = pickle.load(f)
    
    with open(MODELS_DIR / 'config_hibrido_v5.pkl', 'rb') as f:
        config = pickle.load(f)
    
    return {
        'early': modelo_early,
        'mid': modelo_mid,
        'late': modelo_late
    }, config

@st.cache_data
def cargar_datos():
    """Carga context stats"""
    with open(DATA_DIR / 'context_stats_rolling.pkl', 'rb') as f:
        context_stats = pickle.load(f)
    return context_stats

# Cargar
modelos, config = cargar_modelos()
context_stats = cargar_datos()

# ==================== TEAM MAPPING ====================

TEAM_MAPPING = {
    'Aston Martin': 'Aston Martin',
    'Aston Martin F1 Team': 'Aston Martin',
    'Racing Point': 'Aston Martin',
    'Racing Point F1 Team': 'Aston Martin',
    'Force India': 'Aston Martin',
    'Alpine': 'Alpine',
    'Alpine F1 Team': 'Alpine',
    'Renault': 'Alpine',
    'Red Bull Racing': 'Red Bull Racing',
    'Red Bull': 'Red Bull Racing',
    'Racing Bulls': 'AlphaTauri',
    'RB': 'AlphaTauri',
    'AlphaTauri': 'AlphaTauri',
    'Toro Rosso': 'AlphaTauri',
    'Kick Sauber': 'Sauber',
    'Alfa Romeo': 'Sauber',
    'Sauber': 'Sauber',
    'Mercedes': 'Mercedes',
    'Haas': 'Haas',
    'Haas F1 Team': 'Haas',
    'Ferrari': 'Ferrari',
    'McLaren': 'McLaren',
    'Williams': 'Williams',
}

def normalizar_team(team):
    return TEAM_MAPPING.get(team, team)

# Colores por equipo (gamas diferenciadas)
TEAM_COLORS = {
    'Red Bull Racing': ['#1E3A8A', '#3B82F6'],      # Azul oscuro/claro
    'Ferrari': ['#DC2626', '#EF4444'],               # Rojo oscuro/claro
    'Mercedes': ['#059669', '#10B981'],              # Verde oscuro/claro
    'McLaren': ['#EA580C', '#FB923C'],               # Naranja oscuro/claro
    'Aston Martin': ['#047857', '#34D399'],          # Verde esmeralda oscuro/claro
    'Alpine': ['#0284C7', '#38BDF8'],                # Azul cielo oscuro/claro
    'AlphaTauri': ['#4338CA', '#818CF8'],            # Índigo oscuro/claro
    'Sauber': ['#991B1B', '#F87171'],                # Rojo burgundy oscuro/claro
    'Haas': ['#4B5563', '#9CA3AF'],                  # Gris oscuro/claro
    'Williams': ['#1D4ED8', '#60A5FA'],              # Azul real oscuro/claro
}

def obtener_color_piloto(driver, team, df_all):
    """Asigna color según equipo (primer piloto color oscuro, segundo color claro)"""
    if team not in TEAM_COLORS:
        return '#6B7280'  # Gris por defecto
    
    # Obtener pilotos del equipo
    pilotos_equipo = df_all[df_all['team'] == team]['driver'].unique()
    
    # Asignar color oscuro al primero, claro al segundo
    if len(pilotos_equipo) == 1 or driver == pilotos_equipo[0]:
        return TEAM_COLORS[team][0]  # Color oscuro
    else:
        return TEAM_COLORS[team][1]  # Color claro

# ==================== GESTIÓN DE DATOS 2025 ====================

def cargar_datos_2025_locales():
    """Carga datos 2025 desde CSV local"""
    if CACHE_2025_FILE.exists():
        df = pd.read_csv(CACHE_2025_FILE)
        if CACHE_INFO_FILE.exists():
            with open(CACHE_INFO_FILE, 'r') as f:
                info = json.load(f)
        else:
            info = {
                'ultima_actualizacion': 'Desconocida',
                'ultima_ronda': int(df['round'].max())
            }
        return df, info
    else:
        return None, None

def obtener_ultima_carrera_disponible():
    """Verifica última carrera disponible"""
    try:
        cache_dir = Path('cache')
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        
        try:
            schedule = fastf1.get_event_schedule(2025)
            carreras = schedule[schedule['EventFormat'] != 'testing']
            total_carreras = len(carreras)
        except:
            total_carreras = 24
        
        ultima_ronda_disponible = 0
        
        for round_test in range(1, min(10, total_carreras + 1)):
            try:
                session = fastf1.get_session(2025, round_test, 'R')
                if session is not None:
                    ultima_ronda_disponible = round_test
                else:
                    break
            except:
                break
        
        return ultima_ronda_disponible, total_carreras
    
    except Exception as e:
        return 0, 0

def descargar_ronda_2025(round_num):
    """Descarga SOLO resultados de UNA ronda"""
    try:
        cache_dir = Path('cache')
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        
        session = fastf1.get_session(2025, round_num, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        
        results = session.results
        
        if results is None or len(results) == 0:
            return None
        
        datos_ronda = []
        
        for idx, driver_result in results.iterrows():
            driver_code = driver_result.get('Abbreviation')
            if pd.isna(driver_code):
                continue
            
            team_name = driver_result.get('TeamName', 'Unknown')
            position = driver_result.get('Position')
            status = driver_result.get('Status', '')
            
            if pd.isna(position) or position == 0:
                position = 999
                es_abandono = True
            else:
                position = int(position)
                es_abandono = False
            
            if isinstance(status, str):
                if any(x in status.upper() for x in ['DNF', 'RETIRED', 'ACCIDENT', 'COLLISION']):
                    es_abandono = True
                    position = 999
            
            puntos_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
            puntos = puntos_map.get(position, 0) if position < 999 else 0
            
            datos_ronda.append({
                'year': 2025,
                'round': int(round_num),
                'driver': str(driver_code),
                'team': str(team_name),
                'position': position,
                'puntos_carrera': puntos,
                'es_abandono': es_abandono
            })
        
        return pd.DataFrame(datos_ronda)
    
    except Exception as e:
        return None

def actualizar_datos_2025():
    """Actualiza solo nuevas carreras"""
    ultima_disponible, total_carreras = obtener_ultima_carrera_disponible()
    
    if ultima_disponible == 0:
        return None, None, "No hay carreras disponibles"
    
    df_local, info_local = cargar_datos_2025_locales()
    
    if df_local is None:
        st.info(f"📥 Descargando R1-R{ultima_disponible}...")
        todas_las_rondas = []
        progress_bar = st.progress(0)
        
        for i, round_num in enumerate(range(1, ultima_disponible + 1), 1):
            df_ronda = descargar_ronda_2025(round_num)
            if df_ronda is not None:
                todas_las_rondas.append(df_ronda)
            progress_bar.progress(i / ultima_disponible)
        
        progress_bar.empty()
        
        if len(todas_las_rondas) == 0:
            return None, None, "Error descargando"
        
        df_completo = pd.concat(todas_las_rondas, ignore_index=True)
    else:
        ultima_local = info_local['ultima_ronda']
        
        if ultima_disponible == ultima_local:
            return df_local, info_local, "Ya actualizado"
        
        elif ultima_disponible > ultima_local:
            nuevas_rondas = list(range(ultima_local + 1, ultima_disponible + 1))
            st.info(f"📥 Descargando nuevas: {nuevas_rondas}...")
            
            nuevos_datos = []
            progress_bar = st.progress(0)
            
            for i, round_num in enumerate(nuevas_rondas, 1):
                df_ronda = descargar_ronda_2025(round_num)
                if df_ronda is not None:
                    nuevos_datos.append(df_ronda)
                progress_bar.progress(i / len(nuevas_rondas))
            
            progress_bar.empty()
            
            if len(nuevos_datos) == 0:
                return df_local, info_local, "Error descargando nuevas"
            
            df_completo = pd.concat([df_local] + nuevos_datos, ignore_index=True)
        else:
            return df_local, info_local, "Datos actuales"
    
    # Guardar
    df_completo.to_csv(CACHE_2025_FILE, index=False)
    
    info_nueva = {
        'ultima_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ultima_ronda': int(df_completo['round'].max()),
        'total_carreras_temporada': total_carreras
    }
    
    with open(CACHE_INFO_FILE, 'w') as f:
        json.dump(info_nueva, f, indent=2)
    
    return df_completo, info_nueva, f"✅ Actualizado hasta R{info_nueva['ultima_ronda']}"

# Cargar datos 2025
df_2025_local, info_2025 = cargar_datos_2025_locales()
ultima_disponible, total_carreras_2025 = obtener_ultima_carrera_disponible()

# ==================== CÁLCULO DE FEATURES (CON CACHE) ====================

def calcular_todas_las_features(df_año, total_carreras):
    """Calcula features para TODAS las rondas y pilotos"""
    
    year = df_año['year'].iloc[0]
    max_round = int(df_año['round'].max())
    
    todas_las_features = {}
    
    for round_num in range(1, max_round + 1):
        features_ronda = {}
        
        for driver in df_año['driver'].unique():
            df_piloto = df_año[df_año['driver'] == driver]
            df_hasta = df_piloto[df_piloto['round'] <= round_num].copy()
            
            if len(df_hasta) == 0:
                continue
            
            puntos_totales_temporada = total_carreras * 101
            
            df_hasta['linear_points'] = df_hasta['position'].apply(
                lambda p: max(0, 20 - p + 1) if p <= 20 else 0
            )
            
            puntos_acumulados = df_hasta['puntos_carrera'].sum()
            pct_puntos_actual = puntos_acumulados / puntos_totales_temporada
            
            linear_points_acum = df_hasta['linear_points'].sum()
            pct_linear_points = linear_points_acum / (total_carreras * 20)
            
            terminadas = df_hasta[~df_hasta['es_abandono']]
            tendencia_ultimas_3 = terminadas.tail(3)['position'].mean() if len(terminadas) >= 3 else 20.0
            
            progreso_temporada = round_num / total_carreras
            
            team = normalizar_team(df_piloto['team'].iloc[0])
            
            if year in context_stats:
                stats = context_stats[year]
                driver_quality_3y = stats['drivers'].get(driver, 50.0)
                team_avg_pos_3y = stats['teams']['avg_pos_3y'].get(team, 10.0)
                team_trend = stats['teams']['trend'].get(team, 0.0)
            else:
                driver_quality_3y = 50.0
                team_avg_pos_3y = 10.0
                team_trend = 0.0
            
            features_ronda[driver] = {
                'driver': driver,
                'team': team,
                'pct_puntos_actual': pct_puntos_actual,
                'pct_linear_points': pct_linear_points,
                'tendencia_ultimas_3': tendencia_ultimas_3,
                'diff_con_lider_normalizada': 0.0,
                'progreso_temporada': progreso_temporada,
                'driver_quality_3y': driver_quality_3y,
                'team_avg_pos_3y': team_avg_pos_3y,
                'team_trend': team_trend
            }
        
        # Calcular diff_con_lider
        if len(features_ronda) > 0:
            max_puntos = max(f['pct_puntos_actual'] for f in features_ronda.values())
            for driver_features in features_ronda.values():
                driver_features['diff_con_lider_normalizada'] = max_puntos - driver_features['pct_puntos_actual']
        
        todas_las_features[round_num] = features_ronda
    
    return todas_las_features

@st.cache_data(show_spinner=False)
def obtener_features_cacheadas(_df_año, total_carreras):
    """Obtiene features con cache"""
    return calcular_todas_las_features(_df_año, total_carreras)

# ==================== PREDICCIÓN ====================

def obtener_modelo_apropiado(round_num):
    """Selecciona modelo según ronda"""
    if round_num <= 5:
        return modelos['early'], 'EARLY'
    elif round_num <= 12:
        return modelos['mid'], 'MID'
    else:
        return modelos['late'], 'LATE'

def predecir_ronda(features_ronda, round_num):
    """Predice clasificación para una ronda"""
    
    df_pred = pd.DataFrame(features_ronda.values())
    
    modelo_info, fase = obtener_modelo_apropiado(round_num)
    modelo = modelo_info['modelo']
    features_cols = modelo_info['features']
    
    X = df_pred[features_cols]
    df_pred['pct_puntos_pred'] = modelo.predict(X)
    
    df_pred = df_pred.sort_values('pct_puntos_pred', ascending=False).reset_index(drop=True)
    df_pred['posicion_pred'] = range(1, len(df_pred) + 1)
    
    return df_pred, fase

# ==================== INTERFACE ====================

st.title("🏎️ F1 Championship Predictor")
st.markdown("**Predicción de clasificación final - Temporada 2025**")

# ==================== SIDEBAR ====================

st.sidebar.header("⚙️ Configuración")
st.sidebar.markdown("### 📊 Estado de Datos")

if df_2025_local is not None and info_2025 is not None:
    ultima_local = info_2025['ultima_ronda']
    total_temp = info_2025.get('total_carreras_temporada', 24)
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Rondas cargadas", f"{ultima_local}/{total_temp}")
    col2.metric("Progreso", f"{ultima_local/total_temp*100:.0f}%")
    
    st.sidebar.caption(f"📅 Actualizado: {info_2025['ultima_actualizacion']}")
    
    if ultima_disponible > 0:
        if ultima_disponible > ultima_local:
            st.sidebar.warning(f"🆕 Nueva carrera disponible!")
            hay_actualizacion = True
        else:
            hay_actualizacion = False
    else:
        hay_actualizacion = False
else:
    if ultima_disponible > 0:
        st.sidebar.warning(f"⚠️ Sin datos locales")
        st.sidebar.info(f"Disponible: R1-R{ultima_disponible}")
        hay_actualizacion = True
    else:
        st.sidebar.info("⏳ Temporada no iniciada")
        hay_actualizacion = False

if hay_actualizacion:
    if st.sidebar.button("🔄 Actualizar Datos", type="primary", use_container_width=True):
        with st.spinner("Actualizando..."):
            df_2025_local, info_2025, mensaje = actualizar_datos_2025()
        if df_2025_local is not None:
            st.sidebar.success(mensaje)
            st.rerun()
        else:
            st.sidebar.error(mensaje)
else:
    st.sidebar.button("✅ Datos actualizados", disabled=True, use_container_width=True)

# ==================== PREDICCIÓN ====================

st.header("📊 Predicción de Clasificación")

if df_2025_local is None:
    st.warning("⚠️ No hay datos de 2025. Haz clic en 'Actualizar Datos'")
    st.stop()

df_año = df_2025_local
total_carreras = info_2025.get('total_carreras_temporada', 24)
max_round = int(df_año['round'].max())

# Pre-calcular features
with st.spinner("🔄 Preparando datos..."):
    todas_features = obtener_features_cacheadas(df_año, total_carreras)

st.success(f"✅ Listo para predecir")

# Selector de ronda
if max_round > 1:
    round_hasta = st.slider("📍 Selecciona la ronda:", 1, max_round, max_round)
else:
    st.info(f"📍 Ronda {max_round}")
    round_hasta = max_round

# Predecir
if round_hasta in todas_features:
    df_pred, fase_usada = predecir_ronda(todas_features[round_hasta], round_hasta)
    
# Tabla mejorada con estilo Streamlit
    st.subheader(f"🏆 Clasificación Final Predicha (Ronda {round_hasta})")
    
    df_display = df_pred[['posicion_pred', 'driver', 'team']].copy()
    df_display.columns = ['Posición', 'Piloto', 'Equipo']
    
    # Función para colorear filas
    def destacar_posiciones(row):
        pos = row['Posición']
        if pos == 1:
            return ['background-color: #FFD700; font-weight: bold; color: #000000'] * len(row)
        elif pos == 2:
            return ['background-color: #FFD700; font-weight: bold; color: #000000'] * len(row)
        elif pos == 3:
            return ['background-color: #FFD700; font-weight: bold; color: #000000'] * len(row)
        elif pos <= 6:
            return ['background-color: #C0C0C0; font-weight: bold; color: #000000'] * len(row)
        elif pos <= 10:
            return ['background-color: #CD7F32; color: #FFFFFF'] * len(row)
        else:
            return [''] * len(row)
    
    # Aplicar estilo
    styled_df = df_display.style.apply(destacar_posiciones, axis=1)\
        .set_properties(**{
            'text-align': 'left',
            'font-size': '16px',
            'padding': '12px'
        })\
        .set_table_styles([
            {'selector': 'thead th', 'props': [
                ('background', 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('font-size', '17px'),
                ('padding', '15px'),
                ('text-align', 'left')
            ]},
            {'selector': 'tbody tr:hover', 'props': [
                ('background-color', '#f3f4f6')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('width', '100%')
            ]}
        ])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    # Gráfico evolución - TODOS LOS PILOTOS
    st.subheader("📈 Evolución de Predicciones")
    
    evoluciones = {}
    
    for round_num in range(1, max_round + 1):
        if round_num in todas_features:
            df_pred_round, _ = predecir_ronda(todas_features[round_num], round_num)
            for idx, row in df_pred_round.iterrows():
                driver = row['driver']
                if driver not in evoluciones:
                    evoluciones[driver] = {'rondas': [], 'posiciones': [], 'team': row['team']}
                evoluciones[driver]['rondas'].append(round_num)
                evoluciones[driver]['posiciones'].append(row['posicion_pred'])
    
    # Crear gráfico
    fig = go.Figure()
    
    todos_drivers = df_pred['driver'].tolist()
    
    for driver in todos_drivers:
        if driver in evoluciones:
            team = evoluciones[driver]['team']
            color = obtener_color_piloto(driver, team, df_pred)
            
            fig.add_trace(go.Scatter(
                x=evoluciones[driver]['rondas'],
                y=evoluciones[driver]['posiciones'],
                mode='lines+markers',
                name=f"{driver} ({team})",
                line=dict(width=2.5, color=color),
                marker=dict(size=7, color=color),
                hovertemplate=f"<b>{driver}</b><br>Ronda: %{{x}}<br>Posición: %{{y}}<extra></extra>"
            ))
    
    fig.update_layout(
        title=f"<b>Evolución de Posición Predicha - Temporada 2025</b>",
        xaxis_title="<b>Ronda</b>",
        yaxis_title="<b>Posición Predicha</b>",
        height=750,
        yaxis=dict(
            autorange='reversed', 
            dtick=1,
            gridcolor='#e5e7eb',
            range=[0.5, 20.5]
        ),
        xaxis=dict(
            dtick=1,
            gridcolor='#e5e7eb'
        ),
        hovermode='x unified',
        plot_bgcolor='#f9fafb',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#d1d5db",
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"🤖 Modelo: **{fase_usada}** | R1-R5: EARLY | R6-R12: MID | R13+: LATE")

else:
    st.error("No hay datos para esta ronda")

st.markdown("---")
st.caption("🏎️ **F1 Championship Predictor** | Modelos de Machine Learning para predicción de clasificación final")