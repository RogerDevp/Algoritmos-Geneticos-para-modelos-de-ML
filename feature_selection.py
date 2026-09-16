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
# 1. CARGA Y PREPROCESAMIENTO DE DATOS (STUDENTS PERFORMANCE)
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
# 2. CONFIGURACIÓN DEL MODELO BASE Y ALGORITMO GENÉTICO
# ==========================================================
# Red neuronal simplificada a 2 capas ocultas
arquitectura_mlp = (16, 8)
funcion_activacion = "relu"
optimizador = "adam"
tasa_aprendizaje = 0.001
maximo_iteraciones = 1000
semilla_mlp = 42

# Evaluar modelo original con todas las variables
modelo_original = MLPClassifier(
    hidden_layer_sizes=arquitectura_mlp,
    activation=funcion_activacion,
    solver=optimizador,
    learning_rate_init=tasa_aprendizaje,
    max_iter=maximo_iteraciones,
    random_state=semilla_mlp,
    verbose=False
)
modelo_original.fit(X_train_scaled, y_train)
y_pred_original = modelo_original.predict(X_test_scaled)
accuracy_modelo_original_test = accuracy_score(y_test, y_pred_original)

# Configuración del AG
semilla = 42
rng = np.random.default_rng(semilla)

tamano_poblacion = 10
numero_generaciones = 20
tamano_torneo = 3
probabilidad_mutacion = 0.10

# Preparación de datos para AG
X_datos_AG = X_train.copy().apply(pd.to_numeric, errors="coerce")
numero_nan = X_datos_AG.isna().sum().sum()

if numero_nan > 0:
    X_datos_AG = X_datos_AG.fillna(X_datos_AG.median())

X_datos_AG = X_datos_AG.to_numpy(dtype=np.float64)
y_datos_AG = np.asarray(y_train)

escalador_AG = StandardScaler()
X_datos_AG = escalador_AG.fit_transform(X_datos_AG)
X_datos_AG = np.asarray(X_datos_AG, dtype=np.float64)

X_entrenamiento_AG, X_validacion_AG, y_entrenamiento_AG, y_validacion_AG = train_test_split(
    X_datos_AG,
    y_datos_AG,
    test_size=0.20,
    random_state=semilla,
    stratify=y_datos_AG
)

nombres_variables = X_train.columns.tolist()
numero_variables = len(nombres_variables)


# ==========================================================
# 3. IMPRESIÓN - PREPARACIÓN DE DATOS
# ==========================================================
print("=" * 70)
print("PREPARACIÓN DE DATOS PARA FEATURE SELECTION")
print("=" * 70)
print("Valores NaN encontrados:", numero_nan)
print(f"\nDimensiones de X: {X_datos_AG.shape}")
print(f"Dimensiones de y: {y_datos_AG.shape}")
print(f"Tipo de X: {X_datos_AG.dtype}")
print(f"¿Existen NaN?: {np.isnan(X_datos_AG).any()}")
print(f"¿Existen infinitos?: {np.isinf(X_datos_AG).any()}")

print("\nX entrenamiento AG:")
print(X_entrenamiento_AG.shape)
print("X validación AG:")
print(X_validacion_AG.shape)

print(f"\nNúmero total de variables: {numero_variables}")


# ==========================================================
# 4. FUNCIONES DEL ALGORITMO GENÉTICO
# ==========================================================
def crear_genoma():
    genoma = rng.integers(0, 2, size=numero_variables)
    if genoma.sum() == 0:
        genoma[rng.integers(0, numero_variables)] = 1
    return genoma


def genoma_texto(genoma):
    return "".join(map(str, genoma))


def evaluar_genoma(genoma):
    indices_variables = np.where(genoma == 1)[0]
    X_ent_sel = X_entrenamiento_AG[:, indices_variables]
    X_val_sel = X_validacion_AG[:, indices_variables]

    modelo = MLPClassifier(
        hidden_layer_sizes=arquitectura_mlp,
        activation=funcion_activacion,
        solver=optimizador,
        learning_rate_init=tasa_aprendizaje,
        max_iter=maximo_iteraciones,
        random_state=semilla_mlp,
        verbose=False
    )
    modelo.fit(X_ent_sel, y_entrenamiento_AG)
    predicciones = modelo.predict(X_val_sel)
    return accuracy_score(y_validacion_AG, predicciones)


def seleccion_por_torneo(poblacion, fitness):
    participantes = rng.choice(len(poblacion), size=tamano_torneo, replace=False)
    indice_ganador = participantes[np.argmax([fitness[i] for i in participantes])]
    return poblacion[indice_ganador].copy()


def cruzamiento_un_punto(padre_1, padre_2):
    punto = rng.integers(1, numero_variables)
    hijo_1 = np.concatenate([padre_1[:punto], padre_2[punto:]])
    hijo_2 = np.concatenate([padre_2[:punto], padre_1[punto:]])
    return hijo_1, hijo_2


def mutar_genoma(genoma):
    genoma_mutado = genoma.copy()
    for pos in range(numero_variables):
        if rng.random() < probabilidad_mutacion:
            genoma_mutado[pos] = 1 - genoma_mutado[pos]
    if genoma_mutado.sum() == 0:
        genoma_mutado[rng.integers(0, numero_variables)] = 1
    return genoma_mutado


# ==========================================================
# 5. GENERACIÓN 0 - POBLACIÓN INICIAL
# ==========================================================
poblacion_feature_selection = [crear_genoma() for _ in range(tamano_poblacion)]
fitness_accuracy = [evaluar_genoma(genoma) for genoma in poblacion_feature_selection]

print("\n")
print("=" * 70)
print("GENERACIÓN 0 - POBLACIÓN INICIAL")
print("=" * 70)

for numero_individuo, (genoma, accuracy) in enumerate(zip(poblacion_feature_selection, fitness_accuracy), start=1):
    print(
        f"Individuo {numero_individuo:02d} | "
        f"Genoma: {genoma_texto(genoma)} | "
        f"Variables: {genoma.sum():02d} | "
        f"Accuracy: {accuracy:.4f}"
    )


# ==========================================================
# 6. EVOLUCIÓN DEL ALGORITMO GENÉTICO (GENERACIONES 1 EN ADELANTE)
# ==========================================================
historial_mejor_accuracy = []

for numero_generacion in range(1, numero_generaciones + 1):
    fitness_accuracy = [evaluar_genoma(genoma) for genoma in poblacion_feature_selection]

    indice_mejor_individuo = np.argmax(fitness_accuracy)
    mejor_genoma = poblacion_feature_selection[indice_mejor_individuo].copy()
    mejor_accuracy = fitness_accuracy[indice_mejor_individuo]

    historial_mejor_accuracy.append(mejor_accuracy)

    print("\n" + "=" * 70)
    print(f"GENERACIÓN {numero_generacion}")
    print("=" * 70)

    for numero_individuo, (genoma, accuracy) in enumerate(zip(poblacion_feature_selection, fitness_accuracy), start=1):
        print(
            f"Individuo {numero_individuo:02d} | "
            f"Genoma: {genoma_texto(genoma)} | "
            f"Accuracy: {accuracy:.4f}"
        )

    print("-" * 70)
    print(
        f"MEJOR DE LA GENERACIÓN | "
        f"Genoma: {genoma_texto(mejor_genoma)} | "
        f"Accuracy: {mejor_accuracy:.4f}"
    )

    # Elitismo y reproducción
    nueva_poblacion = [mejor_genoma.copy()]

    while len(nueva_poblacion) < tamano_poblacion:
        padre_1 = seleccion_por_torneo(poblacion_feature_selection, fitness_accuracy)
        padre_2 = seleccion_por_torneo(poblacion_feature_selection, fitness_accuracy)

        hijo_1, hijo_2 = cruzamiento_un_punto(padre_1, padre_2)
        hijo_1 = mutar_genoma(hijo_1)
        hijo_2 = mutar_genoma(hijo_2)

        nueva_poblacion.append(hijo_1)
        if len(nueva_poblacion) < tamano_poblacion:
            nueva_poblacion.append(hijo_2)

    poblacion_feature_selection = nueva_poblacion


# ==========================================================
# 7. RESULTADO FINAL Y COMPARACIÓN
# ==========================================================
fitness_final = [evaluar_genoma(genoma) for genoma in poblacion_feature_selection]
indice_mejor_final = np.argmax(fitness_final)
mejor_genoma_final = poblacion_feature_selection[indice_mejor_final].copy()
mejor_accuracy_final = fitness_final[indice_mejor_final]

indices_seleccionados = np.where(mejor_genoma_final == 1)[0]
variables_seleccionadas = [nombres_variables[i] for i in indices_seleccionados]

print("\n")
print("=" * 70)
print("RESULTADO FINAL - FEATURE SELECTION")
print("=" * 70)
print("Mejor genoma:", genoma_texto(mejor_genoma_final))
print(f"Mejor Accuracy de validación: {mejor_accuracy_final:.4f}")
print(f"Número de variables seleccionadas: {len(variables_seleccionadas)}")

print("\nVariables seleccionadas:")
for numero, variable in enumerate(variables_seleccionadas, start=1):
    print(f"{numero:02d}. {variable}")

# Evaluación en TEST
X_train_fs = X_train_scaled[:, indices_seleccionados]
X_test_fs = X_test_scaled[:, indices_seleccionados]

modelo_fs_final = MLPClassifier(
    hidden_layer_sizes=arquitectura_mlp,
    activation=funcion_activacion,
    solver=optimizador,
    learning_rate_init=tasa_aprendizaje,
    max_iter=maximo_iteraciones,
    random_state=semilla_mlp,
    verbose=False
)
modelo_fs_final.fit(X_train_fs, y_train)
predicciones_fs = modelo_fs_final.predict(X_test_fs)
accuracy_feature_selection_test = accuracy_score(y_test, predicciones_fs)

print("\n")
print("=" * 70)
print("COMPARACIÓN FINAL")
print("=" * 70)
print(f"Modelo original - {numero_variables} variables: {accuracy_modelo_original_test:.4f}")
print(f"Feature Selection - {len(variables_seleccionadas)} variables: {accuracy_feature_selection_test:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(range(1, numero_generaciones + 1), historial_mejor_accuracy, marker="o")
plt.xlabel("Generación")
plt.ylabel("Mejor Accuracy")
plt.title("Evolución del Fitness - Feature Selection (Students Performance)")
plt.grid(True)
plt.show()