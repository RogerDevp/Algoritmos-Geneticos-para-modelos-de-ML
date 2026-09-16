import os
import glob
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ==========================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS (DATASET COMPLETO)
# ==========================================================
path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
csv_files = glob.glob(os.path.join(path, "*.csv"))
df = pd.read_csv(csv_files[0])

df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["performance_level"] = pd.qcut(df["average_score"], q=3, labels=[0, 1, 2])

feature_cols = [
    "gender", 
    "race/ethnicity", 
    "parental level of education", 
    "lunch", 
    "test preparation course"
]

X_raw = df[feature_cols]
X = pd.get_dummies(X_raw, drop_first=True)
y = df["performance_level"].astype(int)

nombres_variables = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================================
# 2. DEFINICIÓN DEL ESPACIO DE HIPERPARÁMETROS Y AG
# ==========================================================
# Espacio discreto para categóricos y continuo para la tasa de aprendizaje
opciones_optimizador = ["adam", "sgd", "lbfgs"]
opciones_activacion = ["relu", "tanh", "logistic"]

# Rango continuo para learning rate
lr_min = 0.0001
lr_max = 0.0100

arquitectura_mlp = (16, 8)
maximo_iteraciones = 1000
semilla_mlp = 42

# Modelo base por defecto
modelo_base = MLPClassifier(
    hidden_layer_sizes=arquitectura_mlp,
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=maximo_iteraciones,
    random_state=semilla_mlp,
    verbose=False
)
modelo_base.fit(X_train_scaled, y_train)
y_pred_base = modelo_base.predict(X_test_scaled)
accuracy_modelo_base_test = accuracy_score(y_test, y_pred_base)

# Configuración del AG
semilla = 42
rng = np.random.default_rng(semilla)

tamano_poblacion = 10
numero_generaciones = 20
tamano_torneo = 3
probabilidad_mutacion = 0.20

# Preparación de datos para la validación interna del AG
X_datos_AG = X_train_scaled.copy()
y_datos_AG = np.asarray(y_train)

X_entrenamiento_AG, X_validacion_AG, y_entrenamiento_AG, y_validacion_AG = train_test_split(
    X_datos_AG,
    y_datos_AG,
    test_size=0.20,
    random_state=semilla,
    stratify=y_datos_AG
)


# ==========================================================
# 3. IMPRESIÓN - PREPARACIÓN DE DATOS
# ==========================================================
print("=" * 70)
print("HYPERPARAMETER OPTIMIZATION (MUTACIÓN CONTINUA DE LEARNING RATE)")
print("=" * 70)
print(f"Rango continuo de LR: [{lr_min} - {lr_max}] (Centrado mediante calibración fina)")
print(f"Dimensiones X entrenamiento AG: {X_entrenamiento_AG.shape}")
print(f"Dimensiones X validación AG: {X_validacion_AG.shape}")


# ==========================================================
# 4. FUNCIONES DEL ALGORITMO GENÉTICO
# ==========================================================
def crear_genoma():
    # Inicialización aleatoria cerca de 0.001 (entre 0.0005 y 0.0025)
    lr_inicial = float(rng.uniform(0.0005, 0.0025))
    idx_opt = int(rng.integers(0, len(opciones_optimizador)))
    idx_act = int(rng.integers(0, len(opciones_activacion)))
    return [lr_inicial, idx_opt, idx_act]


def genoma_texto(genoma):
    lr = genoma[0]
    opt = opciones_optimizador[int(genoma[1])]
    act = opciones_activacion[int(genoma[2])]
    return f"lr={lr:.6f} | opt={opt:<5} | act={act:<8}"


def evaluar_genoma(genoma):
    lr = genoma[0]
    opt = opciones_optimizador[int(genoma[1])]
    act = opciones_activacion[int(genoma[2])]

    modelo = MLPClassifier(
        hidden_layer_sizes=arquitectura_mlp,
        activation=act,
        solver=opt,
        learning_rate_init=lr,
        max_iter=maximo_iteraciones,
        random_state=semilla_mlp,
        verbose=False
    )
    try:
        modelo.fit(X_entrenamiento_AG, y_entrenamiento_AG)
        predicciones = modelo.predict(X_validacion_AG)
        return accuracy_score(y_validacion_AG, predicciones)
    except Exception:
        return 0.0


def seleccion_por_torneo(poblacion, fitness):
    participantes = rng.choice(len(poblacion), size=tamano_torneo, replace=False)
    indice_ganador = participantes[np.argmax([fitness[i] for i in participantes])]
    return [poblacion[indice_ganador][0], poblacion[indice_ganador][1], poblacion[indice_ganador][2]]


def cruzamiento_un_punto(padre_1, padre_2):
    punto = rng.integers(1, 3)
    hijo_1 = padre_1[:punto] + padre_2[punto:]
    hijo_2 = padre_2[:punto] + padre_1[punto:]
    return hijo_1, hijo_2


def mutar_genoma(genoma):
    genoma_mutado = list(genoma)
    
    # 1. Mutación por calibración (suma/resta pequeña gaussiana) en learning rate
    if rng.random() < probabilidad_mutacion:
        ruido = rng.normal(loc=0.0, scale=0.0003)  # Ajustes de +-0.0003 aprox.
        nuevo_lr = genoma_mutado[0] + ruido
        genoma_mutado[0] = float(np.clip(nuevo_lr, lr_min, lr_max))

    # 2. Mutación en optimizador
    if rng.random() < probabilidad_mutacion:
        genoma_mutado[1] = int(rng.integers(0, len(opciones_optimizador)))

    # 3. Mutación en función de activación
    if rng.random() < probabilidad_mutacion:
        genoma_mutado[2] = int(rng.integers(0, len(opciones_activacion)))

    return genoma_mutado


# ==========================================================
# 5. GENERACIÓN 0 - POBLACIÓN INICIAL
# ==========================================================
poblacion_hiperparametros = [crear_genoma() for _ in range(tamano_poblacion)]
fitness_accuracy = [evaluar_genoma(genoma) for genoma in poblacion_hiperparametros]

print("\n" + "=" * 70)
print("GENERACIÓN 0 - POBLACIÓN INICIAL")
print("=" * 70)

for numero_individuo, (genoma, accuracy) in enumerate(zip(poblacion_hiperparametros, fitness_accuracy), start=1):
    print(
        f"Individuo {numero_individuo:02d} | "
        f"Configuración: [{genoma_texto(genoma)}] | "
        f"Accuracy: {accuracy:.4f}"
    )


# ==========================================================
# 6. EVOLUCIÓN DEL ALGORITMO GENÉTICO
# ==========================================================
historial_mejor_accuracy = []

for numero_generacion in range(1, numero_generaciones + 1):
    fitness_accuracy = [evaluar_genoma(genoma) for genoma in poblacion_hiperparametros]

    indice_mejor_individuo = np.argmax(fitness_accuracy)
    mejor_genoma = list(poblacion_hiperparametros[indice_mejor_individuo])
    mejor_accuracy = fitness_accuracy[indice_mejor_individuo]

    historial_mejor_accuracy.append(mejor_accuracy)

    print("\n" + "=" * 70)
    print(f"GENERACIÓN {numero_generacion}")
    print("=" * 70)

    for numero_individuo, (genoma, accuracy) in enumerate(zip(poblacion_hiperparametros, fitness_accuracy), start=1):
        print(
            f"Individuo {numero_individuo:02d} | "
            f"Configuración: [{genoma_texto(genoma)}] | "
            f"Accuracy: {accuracy:.4f}"
        )

    print("-" * 70)
    print(
        f"MEJOR DE LA GENERACIÓN | "
        f"Configuración: [{genoma_texto(mejor_genoma)}] | "
        f"Accuracy: {mejor_accuracy:.4f}"
    )

    nueva_poblacion = [list(mejor_genoma)]

    while len(nueva_poblacion) < tamano_poblacion:
        padre_1 = seleccion_por_torneo(poblacion_hiperparametros, fitness_accuracy)
        padre_2 = seleccion_por_torneo(poblacion_hiperparametros, fitness_accuracy)

        hijo_1, hijo_2 = cruzamiento_un_punto(padre_1, padre_2)
        hijo_1 = mutar_genoma(hijo_1)
        hijo_2 = mutar_genoma(hijo_2)

        nueva_poblacion.append(hijo_1)
        if len(nueva_poblacion) < tamano_poblacion:
            nueva_poblacion.append(hijo_2)

    poblacion_hiperparametros = nueva_poblacion


# ==========================================================
# 7. RESULTADO FINAL Y COMPARACIÓN EN TEST
# ==========================================================
fitness_final = [evaluar_genoma(genoma) for genoma in poblacion_hiperparametros]
indice_mejor_final = np.argmax(fitness_final)
mejor_genoma_final = poblacion_hiperparametros[indice_mejor_final]
mejor_accuracy_final = fitness_final[indice_mejor_final]

best_lr = mejor_genoma_final[0]
best_opt = opciones_optimizador[int(mejor_genoma_final[1])]
best_act = opciones_activacion[int(mejor_genoma_final[2])]

print("\n" + "=" * 70)
print("RESULTADO FINAL - HYPERPARAMETER OPTIMIZATION")
print("=" * 70)
print(f"Mejor configuración optimizada encontrándola mediante mutación continua:")
print(f"  - Tasa de aprendizaje (learning_rate_init): {best_lr:.6f}")
print(f"  - Optimizador (solver): {best_opt}")
print(f"  - Función de activación (activation): {best_act}")
print(f"Mejor Accuracy de validación: {mejor_accuracy_final:.4f}")

modelo_opt_final = MLPClassifier(
    hidden_layer_sizes=arquitectura_mlp,
    activation=best_act,
    solver=best_opt,
    learning_rate_init=best_lr,
    max_iter=maximo_iteraciones,
    random_state=semilla_mlp,
    verbose=False
)
modelo_opt_final.fit(X_train_scaled, y_train)
predicciones_opt = modelo_opt_final.predict(X_test_scaled)
accuracy_hiperparametros_test = accuracy_score(y_test, predicciones_opt)

print("\n" + "=" * 70)
print("COMPARACIÓN FINAL EN CONJUNTO DE PRUEBA (TEST)")
print("=" * 70)
print(f"Modelo Base (Default params): {accuracy_modelo_base_test:.4f}")
print(f"Modelo Optimizado (AG Continuo): {accuracy_hiperparametros_test:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(range(1, numero_generaciones + 1), historial_mejor_accuracy, marker="o", color="green")
plt.xlabel("Generación")
plt.ylabel("Mejor Accuracy")
plt.title("Evolución del Fitness - Calibración Continua de LR (AG)")
plt.grid(True)
plt.show()