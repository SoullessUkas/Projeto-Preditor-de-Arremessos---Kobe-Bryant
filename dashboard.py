import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, log_loss, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard de Monitoramento", layout="wide")
st.title("📊 Dashboard de Monitoramento dos Modelos")

# ====================== CLASSIFICAÇÃO - Decision Tree ============================
st.header("🌳 Modelo de Classificação - Decision Tree")
try:
    df_clf = pd.read_parquet("data/processed/predicoes_clf.parquet")
    st.subheader("📌 Resultados - Classificação")
    st.dataframe(df_clf)

    # Métricas
    f1 = f1_score(df_clf["real"], df_clf["prediction_label"])
    logloss = log_loss(df_clf["real"], df_clf["prediction_score"])
    col1, col2 = st.columns(2)
    col1.metric("F1 Score", round(f1, 3))
    col2.metric("Log Loss", round(logloss, 3))

    # Classification report
    st.subheader("📄 Relatório de Classificação")
    report = classification_report(df_clf["real"], df_clf["prediction_label"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # Gráfico das probabilidades
    st.subheader("📈 Distribuição das Probabilidades de Predição")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_clf["prediction_score"], bins=20, kde=True, ax=ax1)
    ax1.set_xlabel("Probabilidade de Acerto")
    ax1.set_ylabel("Frequência")
    st.pyplot(fig1)

except Exception as e:
    st.error(f"❌ Erro ao carregar dados de classificação: {e}")


# ====================== REGRESSÃO - Linear Regression ============================
st.header("📐 Modelo de Regressão - Linear Regression")
try:
    df_reg = pd.read_parquet("data/processed/predicoes_reg.parquet")
    st.subheader("📌 Resultados - Regressão")
    st.dataframe(df_reg)

    # Cálculo das métricas
    mae = mean_absolute_error(df_reg["real"], df_reg["prediction_label"])
    mse = mean_squared_error(df_reg["real"], df_reg["prediction_label"])
    rmse = mean_squared_error(df_reg["real"], df_reg["prediction_label"], squared=False)
    r2 = r2_score(df_reg["real"], df_reg["prediction_label"])

    # Métricas em colunas
    st.subheader("📊 Métricas de Regressão")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", round(mae, 3))
    col2.metric("MSE", round(mse, 3))
    col3.metric("RMSE", round(rmse, 3))
    col4.metric("R²", round(r2, 3))

    # Gráfico de dispersão real vs predito
    st.subheader("📉 Dispersão: Valor Real vs Predito")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="real", y="prediction_label", data=df_reg, ax=ax2)
    ax2.set_xlabel("Valor Real")
    ax2.set_ylabel("Valor Predito")
    ax2.set_title("Dispersão dos Valores")
    st.pyplot(fig2)
 
    # Distribuição do erro
    st.subheader("📊 Distribuição dos Erros")
    df_reg["erro"] = df_reg["real"] - df_reg["prediction_label"]
    fig3, ax3 = plt.subplots()
    sns.histplot(df_reg["erro"], bins=20, kde=True, ax=ax3)
    ax3.set_xlabel("Erro")
    ax3.set_ylabel("Frequência")
    st.pyplot(fig3)

except Exception as e:
    st.error(f"❌ Erro ao carregar dados de regressão: {e}")

# ====================== 🔮 Preditor ao vivo ============================
st.header("🧪 Fazer uma nova predição manual")

with st.form("form_predicao"):
    st.subheader("📥 Informar variáveis de entrada:")
    
    lat = st.number_input("Latitude", value=33.0)
    lon = st.number_input("Longitude", value=-118.0)
    minutos = st.slider("Minutos restantes", 0, 12, 6)
    periodo = st.selectbox("Período do jogo", [1, 2, 3, 4])
    playoffs = st.selectbox("É playoff?", ["Sim", "Não"])
    distancia = st.slider("Distância do arremesso (ft)", 0, 50, 15)

    submitted = st.form_submit_button("🔍 Realizar Predição")

    if submitted:
        try:
           
            entrada = {
                "lat": lat,
                "lon": lon,
                "minutes_remaining": minutos,
                "period": periodo,
                "playoffs": 1 if playoffs == "Sim" else 0,
                "shot_distance": distancia
            }
            df_novo = pd.DataFrame([entrada])

            modelo_path = "outputs/models/DecisionTreeClassifier_clf_final.pkl"
            modelo = joblib.load(modelo_path)
            pred = modelo.predict(df_novo)[0]
            prob = modelo.predict_proba(df_novo)[0][1]

            st.success(f"🎯 Predição: {'Acertou' if pred == 1 else 'Errou'} a cesta")
            st.info(f"Probabilidade de acerto: {prob:.2%}")
        
        except Exception as e:
            st.error(f"Erro ao carregar modelo ou realizar predição: {e}")