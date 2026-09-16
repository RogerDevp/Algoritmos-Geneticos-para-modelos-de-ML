---> python main.py
Dimensiones de X: (1000, 12)

Primeras filas de las características codificadas:
   gender_male  race/ethnicity_group B  ...  lunch_standard  test preparation course_none
0        False                    True  ...            True                          True
1        False                   False  ...            True                         False
2        False                    True  ...            True                          True
3         True                   False  ...           False                          True
4         True                   False  ...            True                          True

[5 rows x 12 columns]

X_train: (800, 12)
X_test : (200, 12)

Distribución de las clases en el conjunto completo:
performance_level
0    336
1    332
2    332
Name: count, dtype: int64

Distribución de las clases en el conjunto de entrenamiento:
performance_level
0    269
2    266
1    265
Name: count, dtype: int64

Distribución de las clases en el conjunto de prueba:
performance_level
1    67
0    67
2    66
Name: count, dtype: int64
C:\Users\Megumi\AppData\Roaming\Python\Python314\site-packages\sklearn\neural_network\_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
  warnings.warn(

MLPClassifier entrenado con éxito.

Reporte de Clasificación:

              precision    recall  f1-score   support

        Bajo       0.44      0.55      0.49        67
       Medio       0.29      0.25      0.27        67
        Alto       0.52      0.44      0.48        66

    accuracy                           0.41       200
   macro avg       0.41      0.42      0.41       200
weighted avg       0.41      0.41      0.41       200



---> python feature_selection.py


==============================================================
PREPARACIÓN DE DATOS PARA FEATURE SELECTION
==============================================================
Valores NaN encontrados: 0

Dimensiones de X: (800, 12)
Dimensiones de y: (800,)
Tipo de X: float64
¿Existen NaN?: False
¿Existen infinitos?: False

X entrenamiento AG:
(640, 12)
X validación AG:
(160, 12)

Número total de variables: 12


==============================================================
GENERACIÓN 0 - POBLACIÓN INICIAL
==============================================================
Individuo 01 | Genoma: 011001010011 | Variables: 06 | Accuracy
Individuo 02 | Genoma: 111110101001 | Variables: 08 | Accuracy
Individuo 03 | Genoma: 110110000110 | Variables: 06 | Accuracy
Individuo 04 | Genoma: 110101100101 | Variables: 07 | Accuracy
Individuo 05 | Genoma: 111000001011 | Variables: 06 | Accuracy
Individuo 06 | Genoma: 110100100010 | Variables: 05 | Accuracy
Individuo 07 | Genoma: 001000111001 | Variables: 05 | Accuracy
Individuo 08 | Genoma: 110011011010 | Variables: 07 | Accuracy
Individuo 09 | Genoma: 011010101111 | Variables: 08 | Accuracy
Individuo 10 | Genoma: 010110110000 | Variables: 05 | Accuracy

==============================================================
GENERACIÓN 1
==============================================================
Individuo 01 | Genoma: 011001010011 | Accuracy: 0.4375
Individuo 02 | Genoma: 111110101001 | Accuracy: 0.3875
Individuo 03 | Genoma: 110110000110 | Accuracy: 0.4000
Individuo 04 | Genoma: 110101100101 | Accuracy: 0.4250
Individuo 05 | Genoma: 111000001011 | Accuracy: 0.4500
Individuo 06 | Genoma: 110100100010 | Accuracy: 0.4688
Individuo 07 | Genoma: 001000111001 | Accuracy: 0.3625
Individuo 08 | Genoma: 110011011010 | Accuracy: 0.3937
Individuo 09 | Genoma: 011010101111 | Accuracy: 0.4188
Individuo 10 | Genoma: 010110110000 | Accuracy: 0.3750
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100100010 | Accuracy: 0.46

==============================================================
GENERACIÓN 2
==============================================================
Individuo 01 | Genoma: 110100100010 | Accuracy: 0.4688
Individuo 02 | Genoma: 111101100011 | Accuracy: 0.4625
Individuo 03 | Genoma: 111000110101 | Accuracy: 0.4313
Individuo 04 | Genoma: 110000001011 | Accuracy: 0.4625
Individuo 05 | Genoma: 101000001011 | Accuracy: 0.4688
Individuo 06 | Genoma: 110100100000 | Accuracy: 0.3812
Individuo 07 | Genoma: 111100100010 | Accuracy: 0.4313
Individuo 08 | Genoma: 011001000110 | Accuracy: 0.4000
Individuo 09 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 10 | Genoma: 011101101111 | Accuracy: 0.4625
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 3
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110101101111 | Accuracy: 0.4500
Individuo 03 | Genoma: 001100110111 | Accuracy: 0.4313
Individuo 04 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 05 | Genoma: 110000110111 | Accuracy: 0.4562
Individuo 06 | Genoma: 110100101010 | Accuracy: 0.4500
Individuo 07 | Genoma: 110101111011 | Accuracy: 0.4313
Individuo 08 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 09 | Genoma: 011101101111 | Accuracy: 0.4625
Individuo 10 | Genoma: 110101100010 | Accuracy: 0.4500
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

======================================================================GENERACIÓN 4
======================================================================Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 111000110111 | Accuracy: 0.4250
Individuo 03 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 04 | Genoma: 110110110111 | Accuracy: 0.4188
Individuo 05 | Genoma: 011101101101 | Accuracy: 0.3812
Individuo 06 | Genoma: 101000111111 | Accuracy: 0.4500
Individuo 07 | Genoma: 111100110111 | Accuracy: 0.4062
Individuo 08 | Genoma: 100101110111 | Accuracy: 0.4562
Individuo 09 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 10 | Genoma: 110101110111 | Accuracy: 0.4437
----------------------------------------------------------------------MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.4813

======================================================================GENERACIÓN 5
======================================================================Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 101111110101 | Accuracy: 0.3750
Individuo 03 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 04 | Genoma: 110101110011 | Accuracy: 0.4625
Individuo 05 | Genoma: 110110110101 | Accuracy: 0.3625
Individuo 06 | Genoma: 010100110111 | Accuracy: 0.4562
Individuo 07 | Genoma: 101000111101 | Accuracy: 0.4375
Individuo 08 | Genoma: 100100110111 | Accuracy: 0.4500
Individuo 09 | Genoma: 111000111101 | Accuracy: 0.4375
Individuo 10 | Genoma: 100100011111 | Accuracy: 0.4250
----------------------------------------------------------------------MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.4813

======================================================================GENERACIÓN 6
======================================================================Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110101100111 | Accuracy: 0.4813
Individuo 03 | Genoma: 110100110011 | Accuracy: 0.4437
Individuo 04 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 05 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 06 | Genoma: 101100100101 | Accuracy: 0.3688
Individuo 07 | Genoma: 010010101101 | Accuracy: 0.4000
Individuo 08 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 09 | Genoma: 111100010010 | Accuracy: 0.4313
Individuo 10 | Genoma: 010100110111 | Accuracy: 0.4562
----------------------------------------------------------------------MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.4813

======================================================================GENERACIÓN 7
======================================================================Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110100101111 | Accuracy: 0.4750
Individuo 03 | Genoma: 110101110001 | Accuracy: 0.4250
Individuo 04 | Genoma: 110101100010 | Accuracy: 0.4500
Individuo 05 | Genoma: 110100100111 | Accuracy: 0.4750
Individuo 06 | Genoma: 110100010011 | Accuracy: 0.4250
Individuo 07 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 08 | Genoma: 100101100101 | Accuracy: 0.4188
Individuo 09 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 10 | Genoma: 110100110111 | Accuracy: 0.4813
----------------------------------------------------------------------MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.4813

======================================================================GENERACIÓN 8
======================================================================Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 03 | Genoma: 110100111111 | Accuracy: 0.4125
Individuo 04 | Genoma: 110101110010 | Accuracy: 0.4000
Individuo 05 | Genoma: 110101111111 | Accuracy: 0.4750
Individuo 06 | Genoma: 110100111010 | Accuracy: 0.4250
Individuo 07 | Genoma: 110101100111 | Accuracy: 0.4813
Individuo 08 | Genoma: 100101110111 | Accuracy: 0.4562
Individuo 09 | Genoma: 110100100111 | Accuracy: 0.4750
Individuo 10 | Genoma: 110100100111 | Accuracy: 0.4750
----------------------------------------------------------------------MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.4813

======================================================================GENERACIÓN 9
======================================================================Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 111100110111 | Accuracy: 0.4062
Individuo 03 | Genoma: 110101100111 | Accuracy: 0.4813
Individuo 04 | Genoma: 110111111101 | Accuracy: 0.3688
Individuo 05 | Genoma: 110100100111 | Accuracy: 0.4750
Individuo 06 | Genoma: 010000100111 | Accuracy: 0.4625
Individuo 07 | Genoma: 010100110111 | Accuracy: 0.4562
Individuo 08 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 09 | Genoma: 110101010111 | Accuracy: 0.4625
Individuo 10 | Genoma: 110010110111 | Accuracy: 0.4500
----------------------------------------------------------------------MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.4813

======================================================================GENERACIÓN 10
======================================================================Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110110110111 | Accuracy: 0.4188
Individuo 03 | Genoma: 111100110111 | Accuracy: 0.4062
Individuo 04 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 05 | Genoma: 110101100011 | Accuracy: 0.4500
Individuo 06 | Genoma: 110000111011 | Accuracy: 0.4562
Individuo 07 | Genoma: 010101110111 | Accuracy: 0.4188
Individuo 08 | Genoma: 110101010111 | Accuracy: 0.4625
Individuo 09 | Genoma: 110101100011 | Accuracy: 0.4500
Individuo 10 | Genoma: 010000000110 | Accuracy: 0.4562
----------------------------------------------------------------------MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.4813

==============================================================
GENERACIÓN 11
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 010101011111 | Accuracy: 0.4313
Individuo 03 | Genoma: 111100110111 | Accuracy: 0.4062
Individuo 04 | Genoma: 110010110011 | Accuracy: 0.4500
Individuo 05 | Genoma: 110100011111 | Accuracy: 0.4375
Individuo 06 | Genoma: 110000111011 | Accuracy: 0.4562
Individuo 07 | Genoma: 110000110111 | Accuracy: 0.4562
Individuo 08 | Genoma: 110000111111 | Accuracy: 0.4375
Individuo 09 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 10 | Genoma: 110000111011 | Accuracy: 0.4562
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 12
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110010110111 | Accuracy: 0.4500
Individuo 03 | Genoma: 110100110011 | Accuracy: 0.4437
Individuo 04 | Genoma: 110000110111 | Accuracy: 0.4562
Individuo 05 | Genoma: 110100110011 | Accuracy: 0.4437
Individuo 06 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 07 | Genoma: 110010101010 | Accuracy: 0.4500
Individuo 08 | Genoma: 110010110011 | Accuracy: 0.4500
Individuo 09 | Genoma: 010010110011 | Accuracy: 0.4750
Individuo 10 | Genoma: 010100110011 | Accuracy: 0.4437
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 13
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 03 | Genoma: 110000010111 | Accuracy: 0.4625
Individuo 04 | Genoma: 011010100001 | Accuracy: 0.4250
Individuo 05 | Genoma: 000010110011 | Accuracy: 0.4625
Individuo 06 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 07 | Genoma: 110010011111 | Accuracy: 0.4375
Individuo 08 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 09 | Genoma: 010010110011 | Accuracy: 0.4750
Individuo 10 | Genoma: 010000111010 | Accuracy: 0.4688
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 14
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110110110111 | Accuracy: 0.4188
Individuo 03 | Genoma: 110010110011 | Accuracy: 0.4500
Individuo 04 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 05 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 06 | Genoma: 110010010111 | Accuracy: 0.4500
Individuo 07 | Genoma: 010000110111 | Accuracy: 0.4625
Individuo 08 | Genoma: 000010111010 | Accuracy: 0.4688
Individuo 09 | Genoma: 010110110011 | Accuracy: 0.4375
Individuo 10 | Genoma: 110000110110 | Accuracy: 0.4500
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 15
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110100100100 | Accuracy: 0.3688
Individuo 03 | Genoma: 000010111011 | Accuracy: 0.4375
Individuo 04 | Genoma: 110010110011 | Accuracy: 0.4500
Individuo 05 | Genoma: 110101110110 | Accuracy: 0.4062
Individuo 06 | Genoma: 110000110011 | Accuracy: 0.4562
Individuo 07 | Genoma: 010001110011 | Accuracy: 0.4313
Individuo 08 | Genoma: 110010101010 | Accuracy: 0.4500
Individuo 09 | Genoma: 000100110111 | Accuracy: 0.4688
Individuo 10 | Genoma: 110100110111 | Accuracy: 0.4813
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 16
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 03 | Genoma: 110000011111 | Accuracy: 0.4500
Individuo 04 | Genoma: 010100110111 | Accuracy: 0.4562
Individuo 05 | Genoma: 100100110111 | Accuracy: 0.4500
Individuo 06 | Genoma: 110100010011 | Accuracy: 0.4250
Individuo 07 | Genoma: 110000110011 | Accuracy: 0.4562
Individuo 08 | Genoma: 110100111011 | Accuracy: 0.4437
Individuo 09 | Genoma: 100000110001 | Accuracy: 0.3937
Individuo 10 | Genoma: 000010111111 | Accuracy: 0.4437
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 17
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 010100100111 | Accuracy: 0.4500
Individuo 03 | Genoma: 110101101111 | Accuracy: 0.4500
Individuo 04 | Genoma: 110000110111 | Accuracy: 0.4562
Individuo 05 | Genoma: 000000110111 | Accuracy: 0.4562
Individuo 06 | Genoma: 011110111110 | Accuracy: 0.4375
Individuo 07 | Genoma: 000100110111 | Accuracy: 0.4688
Individuo 08 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 09 | Genoma: 110111111101 | Accuracy: 0.3688
Individuo 10 | Genoma: 010100110111 | Accuracy: 0.4562
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 18
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 010100110111 | Accuracy: 0.4562
Individuo 03 | Genoma: 110100111111 | Accuracy: 0.4125
Individuo 04 | Genoma: 000100110111 | Accuracy: 0.4688
Individuo 05 | Genoma: 010100111111 | Accuracy: 0.4375
Individuo 06 | Genoma: 110100100111 | Accuracy: 0.4750
Individuo 07 | Genoma: 010000111111 | Accuracy: 0.4250
Individuo 08 | Genoma: 110000110111 | Accuracy: 0.4562
Individuo 09 | Genoma: 110100100111 | Accuracy: 0.4750
Individuo 10 | Genoma: 110100110111 | Accuracy: 0.4813
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 110100110111 | Accuracy: 0.48

==============================================================
GENERACIÓN 19
==============================================================
Individuo 01 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 02 | Genoma: 010100100010 | Accuracy: 0.4875
Individuo 03 | Genoma: 110100110110 | Accuracy: 0.3875
Individuo 04 | Genoma: 100100110111 | Accuracy: 0.4500
Individuo 05 | Genoma: 000100110111 | Accuracy: 0.4688
Individuo 06 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 07 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 08 | Genoma: 000101110111 | Accuracy: 0.4500
Individuo 09 | Genoma: 110100110111 | Accuracy: 0.4813
Individuo 10 | Genoma: 110100000111 | Accuracy: 0.4188
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 010100100010 | Accuracy: 0.48

==============================================================
GENERACIÓN 20
==============================================================
Individuo 01 | Genoma: 010100100010 | Accuracy: 0.4875
Individuo 02 | Genoma: 000101110011 | Accuracy: 0.4625
Individuo 03 | Genoma: 110100100111 | Accuracy: 0.4750
Individuo 04 | Genoma: 110000100010 | Accuracy: 0.4750
Individuo 05 | Genoma: 011100110110 | Accuracy: 0.4500
Individuo 06 | Genoma: 010100110110 | Accuracy: 0.4000
Individuo 07 | Genoma: 110100100000 | Accuracy: 0.3812
Individuo 08 | Genoma: 000111110111 | Accuracy: 0.4688
Individuo 09 | Genoma: 100100100010 | Accuracy: 0.4875
Individuo 10 | Genoma: 110100111111 | Accuracy: 0.4125
--------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Genoma: 010100100010 | Accuracy: 0.48


==============================================================
RESULTADO FINAL - FEATURE SELECTION
==============================================================
Mejor genoma: 010100100010
Mejor Accuracy de validación: 0.4875
Número de variables seleccionadas: 4

Variables seleccionadas:
01. race/ethnicity_group B
02. race/ethnicity_group D
03. parental level of education_high school
04. lunch_standard


==============================================================
COMPARACIÓN FINAL
==============================================================
Modelo original - 12 variables: 0.4150
Feature Selection - 4 variables: 0.4000


---> python hiperparameter_optimization.py

======================================================================
PREPARACIÓN DE DATOS PARA NEUROEVOLUTION (ARQUITECTURA DINÁMICA)
======================================================================
Dataset completo: 12 variables
Arquitectura base de referencia: (16, 8)
Rango de capas permitidas: 1 a 3 capas
Rango de neuronas por capa: 4 a 32 neuronas

======================================================================
GENERACIÓN 0 - POBLACIÓN INICIAL
======================================================================
Individuo 01 | Arquitectura: (20) | Capas: 1       | Accuracy: 0.4375
Individuo 02 | Arquitectura: (15 -> 14) | Capas: 2 | Accuracy: 0.4000
Individuo 03 | Arquitectura: (9 -> 19) | Capas: 2  | Accuracy: 0.4250
Individuo 04 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 05 | Arquitectura: (23 -> 19) | Capas: 2 | Accuracy: 0.4313
Individuo 06 | Arquitectura: (19 -> 20) | Capas: 2 | Accuracy: 0.4688
Individuo 07 | Arquitectura: (10 -> 21) | Capas: 2 | Accuracy: 0.3937
Individuo 08 | Arquitectura: (16) | Capas: 1       | Accuracy: 0.4313
Individuo 09 | Arquitectura: (10) | Capas: 1       | Accuracy: 0.4688
Individuo 10 | Arquitectura: (20 -> 18) | Capas: 2 | Accuracy: 0.4437

======================================================================
GENERACIÓN 1
======================================================================
Individuo 01 | Arquitectura: (20) | Capas: 1       | Accuracy: 0.4375
Individuo 02 | Arquitectura: (15 -> 14) | Capas: 2 | Accuracy: 0.4000
Individuo 03 | Arquitectura: (9 -> 19) | Capas: 2  | Accuracy: 0.4250
Individuo 04 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 05 | Arquitectura: (23 -> 19) | Capas: 2 | Accuracy: 0.4313
Individuo 06 | Arquitectura: (19 -> 20) | Capas: 2 | Accuracy: 0.4688
Individuo 07 | Arquitectura: (10 -> 21) | Capas: 2 | Accuracy: 0.3937
Individuo 08 | Arquitectura: (16) | Capas: 1       | Accuracy: 0.4313
Individuo 09 | Arquitectura: (10) | Capas: 1       | Accuracy: 0.4688
Individuo 10 | Arquitectura: (20 -> 18) | Capas: 2 | Accuracy: 0.4437
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750

======================================================================
GENERACIÓN 2
======================================================================
Individuo 01 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 02 | Arquitectura: (9 -> 20) | Capas: 2  | Accuracy: 0.4313
Individuo 03 | Arquitectura: (19) | Capas: 1       | Accuracy: 0.4188
Individuo 04 | Arquitectura: (11) | Capas: 1       | Accuracy: 0.4313
Individuo 05 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 06 | Arquitectura: (8 -> 19) | Capas: 2  | Accuracy: 0.4125
Individuo 07 | Arquitectura: (23) | Capas: 1       | Accuracy: 0.4437
Individuo 08 | Arquitectura: (20) | Capas: 1       | Accuracy: 0.4375
Individuo 09 | Arquitectura: (7) | Capas: 1        | Accuracy: 0.4375
Individuo 10 | Arquitectura: (15 -> 20) | Capas: 2 | Accuracy: 0.4500
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750

======================================================================
GENERACIÓN 3
======================================================================
Individuo 01 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 02 | Arquitectura: (7) | Capas: 1        | Accuracy: 0.4375
Individuo 03 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 04 | Arquitectura: (15 -> 20 -> 4) | Capas: 3 | Accuracy: 0.4313
Individuo 05 | Arquitectura: (17) | Capas: 1       | Accuracy: 0.4250
Individuo 06 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 07 | Arquitectura: (9 -> 5) | Capas: 2   | Accuracy: 0.4313
Individuo 08 | Arquitectura: (7) | Capas: 1        | Accuracy: 0.4375
Individuo 09 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 10 | Arquitectura: (9 -> 14) | Capas: 2  | Accuracy: 0.4313
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938

======================================================================
GENERACIÓN 4
======================================================================
Individuo 01 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 02 | Arquitectura: (9 -> 9) | Capas: 2   | Accuracy: 0.4625
Individuo 03 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 04 | Arquitectura: (9 -> 5) | Capas: 2   | Accuracy: 0.4313
Individuo 05 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 06 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 07 | Arquitectura: (9) | Capas: 1        | Accuracy: 0.4750
Individuo 08 | Arquitectura: (7 -> 11) | Capas: 2  | Accuracy: 0.4562
Individuo 09 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 10 | Arquitectura: (7) | Capas: 1        | Accuracy: 0.4375
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938

======================================================================
GENERACIÓN 5
======================================================================
Individuo 01 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 02 | Arquitectura: (7) | Capas: 1        | Accuracy: 0.4375
Individuo 03 | Arquitectura: (10 -> 11) | Capas: 2 | Accuracy: 0.4437
Individuo 04 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 05 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 06 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
Individuo 07 | Arquitectura: (10) | Capas: 1       | Accuracy: 0.4688
Individuo 08 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 09 | Arquitectura: (6 -> 4) | Capas: 2   | Accuracy: 0.4938
Individuo 10 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938

======================================================================
GENERACIÓN 6
======================================================================
Individuo 01 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 02 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 03 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 04 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 05 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 06 | Arquitectura: (8 -> 10) | Capas: 2  | Accuracy: 0.4625
Individuo 07 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 08 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
Individuo 09 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
Individuo 10 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938

======================================================================
GENERACIÓN 7
======================================================================
Individuo 01 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 02 | Arquitectura: (6 -> 9) | Capas: 2   | Accuracy: 0.4688
Individuo 03 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 04 | Arquitectura: (4 -> 6) | Capas: 2   | Accuracy: 0.5000
Individuo 05 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 06 | Arquitectura: (7) | Capas: 1        | Accuracy: 0.4375
Individuo 07 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 08 | Arquitectura: (10) | Capas: 1       | Accuracy: 0.4688
Individuo 09 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 10 | Arquitectura: (8 -> 8) | Capas: 2   | Accuracy: 0.4437
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4 -> 6) | Capas: 2   | Accuracy: 0.5000

======================================================================
GENERACIÓN 8
======================================================================
Individuo 01 | Arquitectura: (4 -> 6) | Capas: 2   | Accuracy: 0.5000
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (8 -> 6) | Capas: 2   | Accuracy: 0.4625
Individuo 04 | Arquitectura: (10) | Capas: 1       | Accuracy: 0.4688
Individuo 05 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 06 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 07 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 08 | Arquitectura: (8 -> 5) | Capas: 2   | Accuracy: 0.4562
Individuo 09 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 10 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 9
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 06 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 07 | Arquitectura: (7) | Capas: 1        | Accuracy: 0.4375
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (8 -> 6 -> 8) | Capas: 3 | Accuracy: 0.4375
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 10
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4 -> 14) | Capas: 2  | Accuracy: 0.4875
Individuo 04 | Arquitectura: (4 -> 6) | Capas: 2   | Accuracy: 0.5000
Individuo 05 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
Individuo 06 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 11
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4 -> 6) | Capas: 2   | Accuracy: 0.5000
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 12
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4 -> 6) | Capas: 2   | Accuracy: 0.5000
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 13
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (5) | Capas: 1        | Accuracy: 0.4437
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 14
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4 -> 8) | Capas: 2   | Accuracy: 0.4875
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 15
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4 -> 9) | Capas: 2   | Accuracy: 0.4688
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (4 -> 8) | Capas: 2   | Accuracy: 0.4875
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 16
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 06 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 07 | Arquitectura: (8) | Capas: 1        | Accuracy: 0.4938
Individuo 08 | Arquitectura: (4 -> 7) | Capas: 2   | Accuracy: 0.4813
Individuo 09 | Arquitectura: (4 -> 5) | Capas: 2   | Accuracy: 0.4688
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 17
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (5 -> 4) | Capas: 2   | Accuracy: 0.4313
Individuo 04 | Arquitectura: (5) | Capas: 1        | Accuracy: 0.4437
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (5) | Capas: 1        | Accuracy: 0.4437
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 18
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (6) | Capas: 1        | Accuracy: 0.4750
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 19
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (5) | Capas: 1        | Accuracy: 0.4437
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 07 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (8 -> 13) | Capas: 2  | Accuracy: 0.4562
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
GENERACIÓN 20
======================================================================
Individuo 01 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 02 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 03 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 04 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 05 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 06 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 07 | Arquitectura: (4 -> 4) | Capas: 2   | Accuracy: 0.4813
Individuo 08 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 09 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
Individuo 10 | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062
----------------------------------------------------------------------
MEJOR DE LA GENERACIÓN | Arquitectura: (4) | Capas: 1        | Accuracy: 0.5062

======================================================================
RESULTADO FINAL - NEUROEVOLUTION
======================================================================
Mejor arquitectura encontrada: (4,)
Número de capas ocultas: 1
Mejor Accuracy de validación AG: 0.5062

======================================================================
COMPARACIÓN FINAL EN CONJUNTO DE PRUEBA (TEST)
======================================================================
Modelo Base (Arquitectura (16, 8)): 0.4150
Modelo Optimizado (Arquitectura (4,)): 0.4500
