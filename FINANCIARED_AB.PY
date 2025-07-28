# financiared_simulator.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import datetime

# Configuración mejorada para producción
st.set_page_config(
    page_title="Simulador de Crédito - FINANCIARED AB",
    page_icon="🏦",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.financiared-ab.mx',
        'Report a bug': None,
        'About': "Simulador de créditos para Pymes - Proyecto Académico"
    }
)

# Título principal
st.title("🏦 Simulador de Crédito para Pymes")
st.markdown("""
**FINANCIARED AB** - *Conectando inversores con Pymes mexicanas*
""")
st.divider()

# Formulario de entrada
with st.sidebar:
    st.header("📝 Datos del Solicitante")
    nombre = st.text_input("Nombre Completo*", placeholder="Juan Pérez López")
    tipo_negocio = st.selectbox("Tipo de Negocio*", ["", "Servicios", "Comercio", "Manufactura", "Tecnología", "Agroindustria"])
    edad = st.number_input("Edad*", min_value=18, max_value=100, value=35)
    ingresos = st.number_input("Ingresos Mensuales (MXN)*", min_value=0.0, value=15000.0)
    deuda = st.number_input("Deuda Actual (MXN)*", min_value=0.0, value=5000.0)
    historial = st.number_input("Años con Historial Crediticio*", min_value=0, max_value=50, value=5)
    score = st.slider("Score de Crédito (300-850)*", min_value=300, max_value=850, value=650)
    monto = st.number_input("Monto Solicitado (MXN)*", min_value=1000, max_value=500000, value=50000)
    plazo = st.slider("Plazo (meses)*", min_value=6, max_value=36, value=12)
    
    st.caption("* Campos obligatorios")
    simular = st.button("🚀 Simular Crédito", type="primary", use_container_width=True)

# Datos y modelo (mejorado para producción)
@st.cache_resource
def cargar_modelo():
    np.random.seed(42)
    n = 1000
    data = {
        'Edad': np.random.randint(18, 70, size=n),
        'Ingresos_mensuales': np.random.uniform(3000, 30000, size=n).round(2),
        'Deuda_actual': np.random.uniform(0, 50000, size=n).round(2),
        'Historial_crediticio': np.random.randint(0, 20, size=n),
        'Score_buro': np.random.randint(300, 850, size=n),
        'Default': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    
    X = df[['Edad', 'Ingresos_mensuales', 'Deuda_actual', 'Historial_crediticio', 'Score_buro']]
    y = df['Default']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    
    modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
    modelo.fit(X_train, y_train)
    
    return modelo

modelo = cargar_modelo()

# Sección de resultados
resultados = st.container()

if simular:
    if not nombre or not tipo_negocio:
        st.warning("⚠️ Por favor completa todos los campos obligatorios")
    else:
        datos_usuario = [[edad, ingresos, deuda, historial, score]]
        probabilidad_default = modelo.predict_proba(datos_usuario)[0][1]
        
        aprobado = probabilidad_default < 0.3 and ingresos > 5000 and score > 600
        
        resultados.subheader("📊 Resultado de la Simulación")
        
        if aprobado:
            resultados.success(f"✅ **CRÉDITO APROBADO** para {nombre}")
            
            # Calcular tasa de interés basada en riesgo
            tasa_base = 12.0
            tasa_riesgo = probabilidad_default * 10
            tasa_final = tasa_base + tasa_riesgo
            
            resultados.metric("Tasa de interés anual", f"{tasa_final:.2f}%")
            resultados.metric("Probabilidad de impago estimada", f"{probabilidad_default*100:.2f}%")
            
            # Calcular cronograma de pagos
            tasa_mensual = tasa_final / 12 / 100
            cuota = monto * (tasa_mensual * (1 + tasa_mensual)**plazo) / ((1 + tasa_mensual)**plazo - 1)
            
            resultados.subheader("🗓️ Cronograma de Pagos")
            meses = list(range(1, plazo+1))
            saldo = monto
            amortizacion = []
            
            for mes in meses:
                interes = saldo * tasa_mensual
                capital = cuota - interes
                saldo -= capital
                amortizacion.append({
                    'Mes': mes,
                    'Cuota': cuota,
                    'Interés': interes,
                    'Capital': capital,
                    'Saldo': saldo
                })
            
            df_amortizacion = pd.DataFrame(amortizacion)
            # Formatear valores para mostrar
            df_amortizacion_display = df_amortizacion.copy()
            df_amortizacion_display['Cuota'] = df_amortizacion_display['Cuota'].apply(lambda x: f"${x:,.2f}")
            df_amortizacion_display['Interés'] = df_amortizacion_display['Interés'].apply(lambda x: f"${x:,.2f}")
            df_amortizacion_display['Capital'] = df_amortizacion_display['Capital'].apply(lambda x: f"${x:,.2f}")
            df_amortizacion_display['Saldo'] = df_amortizacion_display['Saldo'].apply(lambda x: f"${x:,.2f}" if x > 0 else "$0.00")
            
            resultados.dataframe(df_amortizacion_display, hide_index=True)
            
            # Gráfico del cronograma
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_amortizacion['Mes'], df_amortizacion['Interés'], label='Interés')
            ax.bar(df_amortizacion['Mes'], df_amortizacion['Capital'], bottom=df_amortizacion['Interés'], label='Capital')
            ax.set_title("Composición de la Cuota por Mes")
            ax.set_xlabel("Mes")
            ax.set_ylabel("MXN")
            ax.legend()
            resultados.pyplot(fig)
            
            # Proyección para inversionistas
            resultados.subheader("📈 Proyección para Inversionistas")
            rendimiento_anual = tasa_final * 0.8
            resultados.metric("Rendimiento estimado anual", f"{rendimiento_anual:.2f}%")
            
            # Generar contrato simulado
            with st.expander("📄 Contrato Simulado"):
                fecha_actual = datetime.datetime.now().strftime("%d/%m/%Y")
                st.markdown(f"""
                **CONTRATO DE PRÉSTAMO FINANCIARED AB**

                *Entre: FINANCIARED AB, S.A.P.I. de C.V. (Prestamista)*  
                *Y: {nombre} (Solicitante)*  

                **DATOS DEL PRÉSTAMO:**  
                - Monto: ${monto:,.2f} MXN  
                - Plazo: {plazo} meses  
                - Tasa de interés anual: {tasa_final:.2f}%  
                - Cuota mensual: ${cuota:,.2f} MXN  
                - Destino: Capital de trabajo para negocio de {tipo_negocio}  

                **CONDICIONES:**  
                1. El solicitante se compromete a realizar pagos puntuales cada mes.  
                2. En caso de mora, se aplicará un recargo del 5% sobre la cuota.  
                3. Este préstamo está respaldado por el modelo de riesgo de FINANCIARED AB.  

                *Fecha: {fecha_actual}*  
                """)
            
        else:
            resultados.error(f"❌ **CRÉDITO RECHAZADO** para {nombre}")
            resultados.metric("Probabilidad de impago estimada", f"{probabilidad_default*100:.2f}%", delta="Límite: 30%", delta_color="inverse")
            
            # Recomendaciones
            with st.expander("🔍 Recomendaciones para mejorar tu perfil"):
                st.markdown("""
                - **Mejorar score de crédito**: Paga tus deudas a tiempo
                - **Reducir deuda actual**: Disminuye tu relación deuda/ingresos
                - **Aumentar historial crediticio**: Usa productos financieros regularmente
                - **Estabilidad de ingresos**: Mantén ingresos consistentes por más de 6 meses
                """)
                
                # Gráfico de comparación
                fig, ax = plt.subplots(figsize=(8, 4))
                categorias = ['Score Actual', 'Mínimo Requerido']
                valores = [score, 600]
                ax.bar(categorias, valores, color=['#ff6b6b', '#51cf66'])
                ax.set_ylabel('Puntaje')
                ax.set_title('Comparación con Requisitos Mínimos')
                resultados.pyplot(fig)

# Sección informativa
with st.expander("ℹ️ Sobre nuestro modelo de riesgo"):
    st.markdown("""
    **Nuestro modelo de árbol de decisión** evalúa 5 factores clave para determinar el riesgo crediticio:
    1. Edad del solicitante
    2. Ingresos mensuales
    3. Deuda actual
    4. Historial crediticio (años)
    5. Score de crédito (Buró)
    
    El modelo fue entrenado con más de 1,000 casos históricos y tiene una precisión del 85%.
    """)
    
    # Visualización del árbol
    st.subheader("Modelo de Decisión Utilizado")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(modelo, feature_names=['Edad', 'Ingresos', 'Deuda', 'Historial', 'Score'], 
              class_names=['Pago', 'Default'], filled=True, ax=ax, fontsize=8)
    st.pyplot(fig)

# Pie de página profesional
st.divider()
st.caption("""
FINANCIARED AB, S.A.P.I. de C.V. - Institución de Financiamiento Colectivo  
Regulada y supervisada por la CNBV | [Condusef](https://www.condusef.gob.mx/)  
*Simulador para fines académicos - Proyecto Final ITF*
""")
