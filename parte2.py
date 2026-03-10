# Transformaciones a los datos y etiquetas

#     ->  Se estandarizan las características de entrada usando 
#         StandardScaler (media 0 y desviación estándar 1) porque las 8 
#         variables tienen escalas muy distintas. Esto es obligatorio para evitar 
#         que los gradientes exploten y para asegurar la convergencia. El scaler 
#         se ajusta únicamente con los datos de entrenamiento para prevenir 
#         filtración de información (data leakage). Las etiquetas (precios en $100k) 
#         se mantienen en su escala original, ya que la capa de salida lineal puede predecir 
#         valores continuos sin transformaciones extra.

# Procedimiento de partición

#     ->  El dataset se dividió usando un 80% para entrenamiento y 20% para prueba, 
#         asegurando suficientes datos (de los 20,640 disponibles) para que la red 
#         aprenda relaciones socioeconómicas complejas, dejando una muestra representativa 
#         para evaluar la generalización. Al ser un problema de regresión, no se usa 
#         estratificación, pero se fijó un random_state para la reproducibilidad.

# Arquitectura de la red neuronal

#      -> La arquitectura cuenta con una entrada de 8 atributos y tres capas 
#         ocultas de tamaño decreciente (64, 32 y 16 neuronas) que comprimen progresivamente 
#         la información para capturar relaciones no lineales complejas. Estas capas usan la 
#         activación ReLU por su eficiencia y prevención del desvanecimiento del gradiente. 
#         La capa de salida consta de una sola neurona con activación lineal, adecuada para 
#         predecir un valor continuo continuo.

# Función de pérdida e hiperparámetros

#      -> Se seleccionó el Error Cuadrático Medio (MSE) como función de pérdida, 
#         ya que penaliza fuertemente los errores grandes, algo crucial al estimar precios. 
#          El optimizador Adam (tasa de aprendizaje de 0.001) garantiza una convergencia estable. 
#         Se usó un batch size de 64 para procesar eficientemente el dataset de mayor tamaño 
#         y se entrenó durante 100 épocas para modelar adecuadamente las complejas relaciones 
#         socioeconómicas. Además, se usó un 10% de validación interna para controlar el 
#         sobreajuste y se incluyó el Error Absoluto Medio (MAE) como métrica adicional, dado 
#         que es directamente interpretable en unidades de precio.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

# Carga del Dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

print("Características:", feature_names)
print(f"Forma del dataset: {X.shape}")
print(f"Precio promedio (target): ${y.mean():.2f}k")
print(f"Precio mínimo: ${y.min():.2f}k | Precio máximo: ${y.max():.2f}k")

# Partición Train y Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"\nTamaño train: {X_train.shape[0]} muestras")
print(f"Tamaño test:  {X_test.shape[0]} muestras")

# Estandarización de los Datos
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nAntes de estandarizar → media: {X_train[:,0].mean():.2f}, std: {X_train[:,0].std():.2f}")
print(f"Después de estandarizar → media: {X_train_sc[:,0].mean():.4f}, std: {X_train_sc[:,0].std():.4f}")

# Arquitectura de la Red Neuronal
model = keras.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
], name="regresor_california_housing")

model.summary()

# Compilación e Hiperparámetros
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Entrenamiento
history = model.fit(
    X_train_sc, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Curvas de Aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train MSE')
axes[0].plot(history.history['val_loss'], label='Val MSE')
axes[0].set_title('MSE durante el entrenamiento')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('MSE')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['mae'],     label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_title('MAE durante el entrenamiento')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('MAE ($100k)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('curvas_aprendizaje_regresion.png', dpi=150)
plt.show()
print("Curvas guardadas en 'curvas_aprendizaje_regresion.png'")

# Métricas de Desempeño
y_pred_train = model.predict(X_train_sc, verbose=0).flatten()
y_pred_test  = model.predict(X_test_sc,  verbose=0).flatten()

def metricas(y_true, y_pred, nombre):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  {nombre}:")
    print(f"    MSE:  {mse:.4f}")
    print(f"    RMSE: {rmse:.4f}  (en $100k)")
    print(f"    MAE:  {mae:.4f}  (en $100k)")
    print(f"    R2:   {r2:.4f}   (1 = perfecto)")

print("\n" + "="*50)
print("MÉTRICAS DE DESEMPEÑO")
print("="*50)
metricas(y_train, y_pred_train, "Train")
metricas(y_test,  y_pred_test,  "Test")

# Scatter Plot: Predicciones vs Valores Reales
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred_test, alpha=0.3, s=10, color='steelblue', label='Predicciones')
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prediccion perfecta')
plt.xlabel('Valor Real (x$100k)', fontsize=12)
plt.ylabel('Prediccion (x$100k)', fontsize=12)
plt.title('Predicciones vs. Valores Reales\nCalifornia Housing', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_predicciones.png', dpi=150)
plt.show()
print("Scatter plot guardado en 'scatter_predicciones.png'")

# Predicciones con Observaciones Nuevas
print("\n" + "="*50)
print("PREDICCIONES CON OBSERVACIONES NUEVAS")
print("="*50)

nuevas_observaciones = np.array([
    [8.5,  20.0, 6.5, 1.1, 800,  2.5, 37.77, -122.42],  # Casa costosa - San Francisco
    [4.2,  35.0, 5.0, 1.0, 1500, 3.0, 34.05, -118.24],  # Casa promedio - L.A. Suburbios
    [1.8,  50.0, 3.5, 1.2, 600,  4.0, 40.00, -122.00],  # Casa economica - Norte CA
])

etiquetas_obs = ["San Francisco (alta gama)", "L.A. Suburbios (media)", "Norte CA (economica)"]

nuevas_sc    = scaler.transform(nuevas_observaciones)
predicciones = model.predict(nuevas_sc, verbose=0).flatten()

for i, (etiq, pred) in enumerate(zip(etiquetas_obs, predicciones)):
    precio_usd = pred * 100_000
    print(f"\n  Observacion {i+1}: {etiq}")
    print(f"  Features -> MedInc={nuevas_observaciones[i,0]}, HouseAge={nuevas_observaciones[i,1]}, "
          f"AveRooms={nuevas_observaciones[i,2]}")
    print(f"  Precio predicho: ${precio_usd:,.0f}  ({pred:.3f} x $100k)")
