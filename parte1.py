# Transformaciones a los datos y etiquetas

#     ->  Se normalizan las imágenes dividiendo por 16 (el valor máximo) para 
#         escalar los píxeles al rango de 0 a 1. Esto evita que valores 
#         grandes dominen el aprendizaje y acelera la convergencia de la red. 
#         Las etiquetas se mantienen como enteros (0 al 9) sin necesidad de aplicar 
#         one-hot encoding, ya que la función de pérdida sparse_categorical_crossentropy 
#         está diseñada para procesar directamente este formato.

# Procedimiento de partición

#     ->  Se utilizó una división del 80% para entrenamiento y 20% para prueba, 
#         garantizando suficientes datos para aprender y un conjunto representativo 
#         para evaluar. Se aplicó un muestreo estratificado (stratify=y) para mantener 
#         la proporción exacta de cada dígito en ambos conjuntos, evitando que la red 
#         se sesgue hacia las clases más frecuentes.

# Arquitectura de la red neuronal

#      -> La red recibe una entrada de 64 atributos y utiliza dos capas 
#         ocultas (128 y 64 neuronas) para aprender desde trazos básicos 
#         hasta patrones abstractos. Ambas emplean la función de activación 
#         ReLU por su eficiencia computacional y para evitar el desvanecimiento 
#         del gradiente. La capa de salida usa 10 neuronas con activación Softmax, 
#         la cual convierte los resultados en probabilidades exactas para clasificar 
#         los 10 posibles dígitos.

# Función de pérdida e hiperparámetros

#      -> Se eligió sparse_categorical_crossentropy por ser la 
#         métrica ideal para clasificación multiclase con etiquetas enteras. 
#         El optimizador Adam (con tasa de aprendizaje de 0.001) asegura una 
#         convergencia rápida y estable. Se definió un batch size de 32 para 
#         equilibrar velocidad y estabilidad de actualización, y se entrenó 
#         por 50 épocas para converger sin llegar al sobreajuste, utilizando un 
#         10% de validación interna para monitorear el progreso.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

# Carga del Dataset
digits = load_digits()
X, y = digits.data, digits.target

print(f"Forma del dataset: {X.shape}")
print(f"Clases: {np.unique(y)}")
print(f"Valor mínimo: {X.min()}, Valor máximo: {X.max()}")

# Transformaciones al Dataset
X_norm = X / 16.0
print(f"\nTras normalización → min: {X_norm.min()}, max: {X_norm.max()}")
print(f"Tipo de etiquetas: {y.dtype}, ejemplo: {y[:5]}")

# Partición Train y Test
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTamaño train: {X_train.shape[0]} muestras")
print(f"Tamaño test:  {X_test.shape[0]} muestras")

# Arquitectura de la Red Neuronal
model = keras.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64,  activation='relu'),
    layers.Dense(10,  activation='softmax')
], name="clasificador_digitos")

model.summary()

# Compilación e Hiperparámetros
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Curvas de Aprendizaje
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Pérdida durante el entrenamiento')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'],     label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Accuracy durante el entrenamiento')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('curvas_aprendizaje_clasificacion.png', dpi=150)
plt.show()
print("Curvas guardadas en 'curvas_aprendizaje_clasificacion.png'")

# Métricas de Desempeño
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss,  test_acc  = model.evaluate(X_test,  y_test,  verbose=0)

print("\n" + "="*50)
print("MÉTRICAS DE DESEMPEÑO")
print("="*50)
print(f"  Train → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
print(f"  Test  → Loss: {test_loss:.4f}  | Accuracy: {test_acc:.4f}")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print("\nReporte de Clasificación (Test):")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Matriz de Confusión - Clasificación de Dígitos', fontsize=14)
plt.ylabel('Etiqueta Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=150)
plt.show()
print("Matriz de confusión guardada en 'matriz_confusion.png'")

# Ejemplos Bien y Mal Clasificados
correct_idx   = np.where(y_pred == y_test)[0]
incorrect_idx = np.where(y_pred != y_test)[0]

def plot_examples(indices, X_data, y_true, y_pred_data, title, filename):
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for i, idx in enumerate(indices[:5]):
        img = X_data[idx].reshape(8, 8)
        axes[i].imshow(img, cmap='gray_r')
        axes[i].set_title(f"Real: {y_true[idx]}\nPred: {y_pred_data[idx]}", fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Figura guardada en '{filename}'")

plot_examples(correct_idx, X_test, y_test, y_pred,
              "5 Ejemplos Bien Clasificados",
              "bien_clasificados.png")

if len(incorrect_idx) > 0:
    plot_examples(incorrect_idx, X_test, y_test, y_pred,
                  f"{min(5, len(incorrect_idx))} Ejemplos Mal Clasificados",
                  "mal_clasificados.png")
else:
    print("No hubo errores — modelo perfecto!")