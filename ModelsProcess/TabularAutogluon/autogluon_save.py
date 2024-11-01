import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suprimir las advertencias

from autogluon.tabular import TabularPredictor
import pandas as pd

# Asegúrate de tener instaladas las dependencias para leer archivos Parquet
# Puedes instalarlas usando: pip install pyarrow

# Cargar el dataset desde un archivo .parquet
df = pd.read_parquet('data/Autogluon_train_80.parquet')

label = 'SleepStage'

# Configuración de los hiperparámetros (opcional)
'''hyperparameters = {
    'GBM': {
        'num_boost_round': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
    },
    'CAT': {
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.03,
    },
}'''

prests = 'best_quality'
hyperparameters = 'default'
time_limit = 28800  # Puedes aumentar este tiempo si es necesario

# Entrenar el modelo con Autogluon
predictor = TabularPredictor(label=label, path='out/autogluon/model/')
predictor.fit(
    df,
    presets=prests,
    time_limit=time_limit,
    hyperparameters=hyperparameters,
    ag_args_fit={
        'ag.max_memory_usage_ratio': 1.5,
    },
    # ag_args_ensemble=dict(fold_fitting_strategy='sequential_local')
)

# Mostrar el leaderboard
leaderboard = predictor.leaderboard(df, silent=True)
print(leaderboard)
