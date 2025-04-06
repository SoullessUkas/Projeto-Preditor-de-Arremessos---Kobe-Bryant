import mlflow
import pandas as pd
from pycaret.classification import load_model, predict_model
from sklearn.metrics import log_loss, f1_score

df_prod = pd.read_parquet("data/raw/dataset_kobe_prod.parquet")
df_prod_filtered = df_prod[[
    "lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance", "shot_made_flag"
]].dropna()

# CLASSIFICAÇÃO - Decision Tree
with mlflow.start_run(run_name="PipelineAplicacao"):
    modelo = load_model("outputs/models/DecisionTreeClassifier_clf_final")

    preds = predict_model(modelo, data=df_prod_filtered)
    df_resultado = pd.DataFrame({
        "real": df_prod_filtered["shot_made_flag"].values,
        "prediction_label": preds["prediction_label"],
        "prediction_score": preds["prediction_score"]
    })

    df_resultado.to_parquet("data/processed/predicoes_clf.parquet", index=False)
    mlflow.log_artifact("data/processed/predicoes_clf.parquet")

    logloss = log_loss(df_resultado["real"], df_resultado["prediction_score"])
    f1 = f1_score(df_resultado["real"], df_resultado["prediction_label"])
    
    mlflow.log_metric("log_loss_producao", logloss)
    mlflow.log_metric("f1_score_producao", f1)

    print("✅ Aplicação de classificação (Decision Tree) executada")

# ======================== REGRESSÃO ==============================
from pycaret.regression import load_model as load_model_reg, predict_model as predict_model_reg
from sklearn.metrics import mean_squared_error, r2_score

df_prod_reg = df_prod_filtered.copy()
df_prod_reg["target_reg"] = df_prod_filtered["shot_made_flag"].astype(float) + 0.1

with mlflow.start_run(run_name="PipelineAplicacao"):
    modelo_reg = load_model_reg("outputs/models/LinearRegression_reg_final")

    preds_reg = predict_model_reg(modelo_reg, data=df_prod_reg)
    df_resultado_reg = pd.DataFrame({
        "real": df_prod_reg["target_reg"].values,
        "prediction_label": preds_reg["prediction_label"],
        "prediction_score": preds_reg["prediction_label"]  # mesma coluna, para manter compatível
    })

    df_resultado_reg.to_parquet("data/processed/predicoes_reg.parquet", index=False)
    mlflow.log_artifact("data/processed/predicoes_reg.parquet")

    rmse = mean_squared_error(df_resultado_reg["real"], df_resultado_reg["prediction_label"], squared=False)
    r2 = r2_score(df_resultado_reg["real"], df_resultado_reg["prediction_label"])
    
    mlflow.log_metric("rmse_producao", rmse)
    mlflow.log_metric("r2_score_producao", r2)

    print("✅ Aplicação de regressão (Linear Regression) executada")
