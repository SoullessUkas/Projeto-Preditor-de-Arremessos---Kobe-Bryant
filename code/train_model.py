import mlflow
import pandas as pd
from pycaret.classification import setup as setup_clf, compare_models as compare_models_clf, save_model as save_model_clf, pull as pull_clf, create_model,tune_model
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, save_model as save_model_reg, pull as pull_reg
from sklearn.metrics import log_loss, f1_score, mean_squared_error, r2_score

# Load datasets
df_train = pd.read_parquet("data/processed/base_train.parquet")
df_test = pd.read_parquet("data/processed/base_test.parquet")

# CLASSIFICATION
X_test_clf = df_test.drop("shot_made_flag", axis=1)
y_test_clf = df_test["shot_made_flag"]

# PyCaret Setup - CLASSIFICATION
setup_clf(
    data=df_train,
    target="shot_made_flag",
    session_id=123,
    verbose=False,
    log_experiment=False
)

with mlflow.start_run(run_name="Treinamento_ArvoreDecisao_Clf"):
    # Treina modelo de árvore de decisão
    model = create_model("dt")  # <- decision tree
    model = tune_model(model)   # <- opcional: ajusta hiperparâmetros

    model_name = model.__class__.__name__
    preds_proba = model.predict_proba(X_test_clf)[:, 1]
    preds_label = model.predict(X_test_clf)

    loss = log_loss(y_test_clf, preds_proba)
    f1 = f1_score(y_test_clf, preds_label)

    mlflow.log_metric("log_loss", loss)
    mlflow.log_metric("f1_score", f1)
    print("Modelo de classificação:", model_name)

    # Salva modelo
    save_model_clf(model, f"outputs/models/{model_name}_clf_final")
    save_model_clf(model, f"mlruns/models/{model_name}_clf_final")
    mlflow.sklearn.log_model(model, f"{model_name}_clf")
# REGRESSION
# Simulação de uma target contínua com base no original (substitua por sua variável real)
df_train_reg = df_train.copy()
df_test_reg = df_test.copy()
df_train_reg["target_reg"] = df_train_reg["shot_made_flag"].astype(float) + 0.1  # exemplo
df_test_reg["target_reg"] = df_test_reg["shot_made_flag"].astype(float) + 0.1

X_test_reg = df_test_reg.drop("target_reg", axis=1)
y_test_reg = df_test_reg["target_reg"]

# PyCaret Setup - REGRESSION
setup_reg(
    data=df_train_reg,
    target="target_reg",
    session_id=123,
    verbose=False,
    log_experiment=False
)

with mlflow.start_run(run_name="Treinamento_Regressao"):
    best_model_reg = compare_models_reg()

    model_name = best_model_reg.__class__.__name__
    preds = best_model_reg.predict(X_test_reg)

    rmse = mean_squared_error(y_test_reg, preds, squared=False)
    r2 = r2_score(y_test_reg, preds)

    mlflow.log_metric(f"rmse_{model_name}", rmse)
    mlflow.log_metric(f"r2_{model_name}", r2)
    print("Modelo de regressão:", model_name)

    # Salva predições para o dashboard
    df_pred_reg = pd.DataFrame({
        "real": y_test_reg.values,
        "prediction": preds
    })
    df_pred_reg.to_parquet("data/processed/predicoes_reg.parquet", index=False)

    save_model_reg(best_model_reg, f"outputs/models/{model_name}_reg_final")
    save_model_reg(best_model_reg, f"mlruns/models/{model_name}_reg_final")
    mlflow.sklearn.log_model(best_model_reg, f"{model_name}_reg")


print("✅ Modelos de classificação e regressão treinados e logados no MLflow")
