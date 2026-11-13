# Ejercicio 01: entrena tu primer modelo y lee sus cinco métricas

Entrenas el modelo de práctica una vez, cuentas a mano cuántos números aprende y conviertes su exactitud en aciertos. Con eso aprendes qué mide cada número que devuelve un entrenamiento, y esa lectura te sirve en todos los ejercicios que siguen.

## Antes de empezar

Como este es el primer ejercicio, no necesitas ninguno anterior. Sí necesitas el taller instalado y una tarjeta de video NVIDIA con CUDA: el [README de los ejercicios](../README.md) te explica cómo instalarlo.

La idea nueva es entrenar un modelo con `juguete.entrenar` y leer lo que devuelve. El modelo de práctica es una red neuronal pequeña que aprende a separar datos en dos clases. En cada dato, la clase depende de tres columnas que sí informan, y otras diez columnas solo traen ruido. Tú le pasas dos cosas:

- **Una config**: un diccionario con las perillas del modelo, que se llaman hiperparámetros. Son números como el ancho de la red o las épocas de entrenamiento, y los eliges tú antes de entrenar porque el modelo no los aprende solo. Las perillas que no escribes quedan en su valor de fábrica.
- **Unas semillas**: números que fijan el azar. Con las mismas semillas, repites exactamente el mismo entrenamiento.

A cambio recibes un diccionario con cinco métricas:

| Métrica | Qué mide |
|---|---|
| `exactitud_val` | La fracción de datos de validación que el modelo acierta. Esos datos no sirvieron para aprender. |
| `exactitud_verdadera` | La misma fracción, medida con una cantidad enorme de datos nuevos. Solo el problema de práctica puede calcularla. |
| `n_parametros` | Cuántos números (pesos) aprendió la red. |
| `epocas` | Cuántas veces vio el modelo todos los datos de entrenamiento. |
| `segundos` | Lo que tardó el entrenamiento. |

## Lo que vas a hacer

Abre `ejercicio.py` y busca los tres comentarios `# TU TURNO`. Cada uno marca un hueco que completas en este orden:

1. **Entrena una vez.** Completa `entrenar_una_vez`: entrena el modelo con la config y las semillas que recibe, y devuelve el diccionario de métricas.
2. **Convierte la exactitud en aciertos.** Completa `aciertos_en_validacion`: calcula cuántos datos de validación acertó el modelo a partir de su exactitud y del total de datos de validación. Devuelve un número entero.
3. **Cuenta los pesos de la red.** Completa `contar_parametros`: suma los pesos y los sesgos de cada capa de la red, y comprueba que tu cuenta coincide con `n_parametros`.

## Pistas

1. **Paso 1.** `juguete.entrenar(config, semillas)` ya hace todo el trabajo y recibe los mismos dos argumentos que recibe tu función. Tu parte es llamarla y devolver lo que te entrega.
2. **Paso 2.** La exactitud es una fracción: aciertos entre total. Para recuperar los aciertos, multiplica por el total. Ten cuidado con el tipo de resultado: `round` y `int` convierten a entero de maneras distintas, y la computadora guarda los decimales en binario, así que `0.29 * 100` no da exactamente 29.
3. **Paso 3.** Cuenta capa por capa. Una capa que recibe `n` números y produce `m` números aprende `n × m` pesos y `m` sesgos. La red tiene `n_capas` capas ocultas del mismo ancho y una capa final que produce un solo número. La primera capa recibe las columnas informativas más las de ruido.

## Comprueba tu trabajo

Corre el verificador desde la raíz del repo:

```bash
uv run python ejercicios/verificar.py 01
```

Cada revisión sale como `Bien: ...` o `Te falta: ...`, con lo que puedes hacer para arreglarla. Mientras un hueco siga sin completar, el verificador dice `Te falta completar`. Cuando todas las revisiones pasen, corre tu programa y lee lo que imprime:

```bash
uv run python ejercicios/01_primer_entrenamiento/ejercicio.py
```

Si te atoras después de leer las pistas, compara con `solucion.py`. Ahí cada comentario explica el porqué de la respuesta.

## Qué observar

Con tu programa funcionando, contesta estas preguntas mirando lo que imprime:

- Corre el programa dos veces seguidas. ¿Qué métricas cambian y cuáles se quedan igual? ¿Por qué crees que pasa eso si las semillas son las mismas?
- Compara `exactitud_val` con `exactitud_verdadera`. ¿Cuál es mayor? En tu tesis no existe la segunda, así que ¿en cuál de las dos tendrías que confiar?
- Las dos clases del problema aparecen con la misma frecuencia, así que adivinar al azar da una exactitud cercana a 0.5. ¿Qué tan lejos de 0.5 queda tu modelo? ¿Aprendió algo?
- Cambia `ancho` de 64 a 16 en `CONFIG` y vuelve a correr. ¿Cuántos pesos tiene ahora la red? ¿Coincide con tu cuenta? ¿Qué le pasó a `exactitud_verdadera`?

## Siguiente ejercicio

[Ejercicio 02: el azar importa](../02_el_azar_importa/README.md). Repites este entrenamiento con varias semillas y mides cuánto cambia el resultado solo por el azar. Si el enlace todavía no abre, ese ejercicio aún no está en el repo: vuelve al [índice de ejercicios](../README.md).
