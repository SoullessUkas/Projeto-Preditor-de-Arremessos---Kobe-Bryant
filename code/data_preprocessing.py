import os
import pandas as pd
import mlflow

# Caminhos de entrada e saída
RAW_PATHS = [
    "data/raw/dataset_kobe_dev.parquet",
    "data/raw/dataset_kobe_prod.parquet"
]
OUTPUT_PATH = "data/processed/data_filtered.parquet"

# Colunas de interesse
FEATURES = [
    "lat",
    "lon",
    "minutes_remaining",
    "period",
    "playoffs",
    "shot_distance",
    "shot_made_flag"
]

def load_and_prepare_data(paths):
    dfs = [pd.read_parquet(path) for path in paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    df_filtered = combined_df[FEATURES].dropna()
    return df_filtered

if __name__ == "__main__":
    with mlflow.start_run(run_name="PreparacaoDados"):
        mlflow.set_tag("stage", "data_processing")
        mlflow.set_tag("author", "Lucas")
        
        # Carrega, filtra e salva
        df = load_and_prepare_data(RAW_PATHS)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)

        # Log do MLflow
        mlflow.log_param("input_files", RAW_PATHS)
        mlflow.log_param("output_file", OUTPUT_PATH)
        mlflow.log_param("num_rows_final", len(df))
        mlflow.log_artifact(OUTPUT_PATH)

        print(f"✅ Dados processados e salvos em {OUTPUT_PATH}")

from sklearn.model_selection import train_test_split

# Separar target
X = df.drop("shot_made_flag", axis=1)
y = df["shot_made_flag"]

# Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Rejunta para salvar
df_train = X_train.copy()
df_train["shot_made_flag"] = y_train

df_test = X_test.copy()
df_test["shot_made_flag"] = y_test

# Salva
df_train.to_parquet("data/processed/base_train.parquet", index=False)
df_test.to_parquet("data/processed/base_test.parquet", index=False)

# Log no MLflow
mlflow.log_param("test_size", 0.2)
mlflow.log_metric("train_size", len(df_train))
mlflow.log_metric("test_size", len(df_test))
