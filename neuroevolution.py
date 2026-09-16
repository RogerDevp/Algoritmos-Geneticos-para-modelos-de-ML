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
# 2. DEFINICIÓN DEL ESPACIO DE NEUROEVOLUCIÓN Y AG
# ==========================================================
# Hiperparámetros fijos según la base del main
activacion_fija = "relu"
optimizador_fijo = "adam"
learning_rate_fijo = 0.001
maximo_iteraciones = 1000
semilla_mlp = 42

# Restricciones para mantener eficiencia computacional
min_capas = 1
max_capas = 3
min_neuronas = 4
max_neuronas = 32

# Modelo base por defecto (16, 8)
arquitectura_inicial = (16, 8)
modelo_base = MLPClassifier(
    hidden_layer_sizes=arquitectura_inicial,
    activation=activacion_fija,
    solver=optimizador_fijo,
    learning_rate_init=learning_rate_fijo,
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
probabilidad_mutacion = 0.25

# Datos para validación interna del AG
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
print("PREPARACIÓN DE DATOS PARA NEUROEVOLUTION (ARQUITECTURA DINÁMICA)")
print("=" * 70)
print(f"Dataset completo: {len(nombres_variables)} variables")
print(f"Arquitectura base de referencia: {arquitectura_inicial}")
print(f"Rango de capas permitidas: {min_capas} a {max_capas} capas")
print(f"Rango de neuronas por capa: {min_neuronas} a {max_neuronas} neuronas")


# ==========================================================
# 4. FUNCIONES DE NEUROEVOLUCIÓN
# ==========================================================
def crear_genoma():
    # Inicialización cercana a la arquitectura base (16, 8) con variaciones aleatorias
    num_capas = rng.integers(1, 3)  # 1 o 2 capas al inicio
    genoma = [int(rng.integers(8, 24)) for _ in range(num_capas)]
    return genoma


def genoma_texto(genoma):
    capas_str = " -> ".join(map(str, genoma))
    return f"Arquitectura: ({capas_str}) | Capas: {len(genoma)}"


def evaluar_genoma(genoma):
    arquitectura = tuple(genoma)
    modelo = MLPClassifier(
        hidden_layer_sizes=arquitectura,
        activation=activacion_fija,
        solver=optimizador_fijo,
        learning_rate_init=learning_rate_fijo,
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
    return list(poblacion[indice_ganador])


def cruzamiento(padre_1, padre_2):
    # Cruzamiento adaptativo según la longitud de los padres
    min_len = min(len(padre_1), len(padre_2))
    punto = rng.integers(1, min_len + 1) if min_len > 1 else 1

    hijo_1 = padre_1[:punto] + padre_2[punto:]
    hijo_2 = padre_2[:punto] + padre_1[punto:]

    # Asegurar restricciones de longitud
    hijo_1 = hijo_1[:max_capas] if hijo_1 else [16]
    hijo_2 = hijo_2[:max_capas] if hijo_2 else [16]

    return hijo_1, hijo_2


def mutar_genoma(genoma):
    genoma_mutado = list(genoma)

    # 1. Mutar número de neuronas mediante ajuste relativo (+/- sumas/restas)
    for i in range(len(genoma_mutado)):
        if rng.random() < probabilidad_mutacion:
            delta = int(rng.choice([-4, -2, -1, 1, 2, 4]))
            nueva_cant = genoma_mutado[i] + delta
            genoma_mutado[i] = int(np.clip(nueva_cant, min_neuronas, max_neuronas))

    # 2. Mutar estructura: agregar o eliminar capa oculta (hasta 3 capas)
    if rng.random() < probabilidad_mutacion:
        opcion_estructura = rng.choice(["agregar", "eliminar"])

        if opcion_estructura == "agregar" and len(genoma_mutado) < max_capas:
            # Insertar nueva capa con neuronas pequeñas/medianas
            nueva_capa = int(rng.integers(4, 16))
            genoma_mutado.append(nueva_capa)

        elif opcion_estructura == "eliminar" and len(genoma_mutado) > min_capas:
            genoma_mutado.pop()

    return genoma_mutado


# ==========================================================
# 5. GENERACIÓN 0 - POBLACIÓN INICIAL
# ==========================================================
poblacion_arquitecturas = [crear_genoma() for _ in range(tamano_poblacion)]
fitness_accuracy = [evaluar_genoma(genoma) for genoma in poblacion_arquitecturas]

print("\n" + "=" * 70)
print("GENERACIÓN 0 - POBLACIÓN INICIAL")
print("=" * 70)

for numero_individuo, (genoma, accuracy) in enumerate(zip(poblacion_arquitecturas, fitness_accuracy), start=1):
    print(
        f"Individuo {numero_individuo:02d} | "
        f"{genoma_texto(genoma):<35} | "
        f"Accuracy: {accuracy:.4f}"
    )


# ==========================================================
# 6. EVOLUCIÓN DEL ALGORITMO GENÉTICO
# ==========================================================
historial_mejor_accuracy = []

for numero_generacion in range(1, numero_generaciones + 1):
    fitness_accuracy = [evaluar_genoma(genoma) for genoma in poblacion_arquitecturas]

    indice_mejor_individuo = np.argmax(fitness_accuracy)
    mejor_genoma = list(poblacion_arquitecturas[indice_mejor_individuo])
    mejor_accuracy = fitness_accuracy[indice_mejor_individuo]

    historial_mejor_accuracy.append(mejor_accuracy)

    print("\n" + "=" * 70)
    print(f"GENERACIÓN {numero_generacion}")
    print("=" * 70)

    for numero_individuo, (genoma, accuracy) in enumerate(zip(poblacion_arquitecturas, fitness_accuracy), start=1):
        print(
            f"Individuo {numero_individuo:02d} | "
            f"{genoma_texto(genoma):<35} | "
            f"Accuracy: {accuracy:.4f}"
        )

    print("-" * 70)
    print(
        f"MEJOR DE LA GENERACIÓN | "
        f"{genoma_texto(mejor_genoma):<35} | "
        f"Accuracy: {mejor_accuracy:.4f}"
    )

    nueva_poblacion = [list(mejor_genoma)]

    while len(nueva_poblacion) < tamano_poblacion:
        padre_1 = seleccion_por_torneo(poblacion_arquitecturas, fitness_accuracy)
        padre_2 = seleccion_por_torneo(poblacion_arquitecturas, fitness_accuracy)

        hijo_1, hijo_2 = cruzamiento(padre_1, padre_2)
        hijo_1 = mutar_genoma(hijo_1)
        hijo_2 = mutar_genoma(hijo_2)

        nueva_poblacion.append(hijo_1)
        if len(nueva_poblacion) < tamano_poblacion:
            nueva_poblacion.append(hijo_2)

    poblacion_arquitecturas = nueva_poblacion


# ==========================================================
# 7. RESULTADO FINAL Y COMPARACIÓN EN TEST
# ==========================================================
fitness_final = [evaluar_genoma(genoma) for genoma in poblacion_arquitecturas]
indice_mejor_final = np.argmax(fitness_final)
mejor_genoma_final = poblacion_arquitecturas[indice_mejor_final]
mejor_accuracy_final = fitness_final[indice_mejor_final]
mejor_arquitectura_tuple = tuple(mejor_genoma_final)

print("\n" + "=" * 70)
print("RESULTADO FINAL - NEUROEVOLUTION")
print("=" * 70)
print(f"Mejor arquitectura encontrada: {mejor_arquitectura_tuple}")
print(f"Número de capas ocultas: {len(mejor_arquitectura_tuple)}")
print(f"Mejor Accuracy de validación AG: {mejor_accuracy_final:.4f}")

# Evaluación final en TEST
modelo_opt_final = MLPClassifier(
    hidden_layer_sizes=mejor_arquitectura_tuple,
    activation=activacion_fija,
    solver=optimizador_fijo,
    learning_rate_init=learning_rate_fijo,
    max_iter=maximo_iteraciones,
    random_state=semilla_mlp,
    verbose=False
)
modelo_opt_final.fit(X_train_scaled, y_train)
predicciones_opt = modelo_opt_final.predict(X_test_scaled)
accuracy_neuroevolution_test = accuracy_score(y_test, predicciones_opt)

print("\n" + "=" * 70)
print("COMPARACIÓN FINAL EN CONJUNTO DE PRUEBA (TEST)")
print("=" * 70)
print(f"Modelo Base (Arquitectura {arquitectura_inicial}): {accuracy_modelo_base_test:.4f}")
print(f"Modelo Optimizado (Arquitectura {mejor_arquitectura_tuple}): {accuracy_neuroevolution_test:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(range(1, numero_generaciones + 1), historial_mejor_accuracy, marker="o", color="purple")
plt.xlabel("Generación")
plt.ylabel("Mejor Accuracy")
plt.title("Evolución del Fitness - Neuroevolution (Estructura y Neuronas)")
plt.grid(True)
plt.show()