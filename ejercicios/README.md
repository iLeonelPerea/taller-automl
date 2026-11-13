# Ejercicios: escribe tú el código para buscar los mejores ajustes de un modelo

Aquí escribes código para aprender a buscar los mejores ajustes de un modelo, empezando por lo más básico y subiendo la dificultad poco a poco. Cada ejercicio te da un archivo con huecos, tú lo completas y un verificador te dice qué quedó bien y qué te falta. Al final conectas tu propio modelo a una búsqueda.

## En qué se distinguen de las lecciones de `pasos/`

Las lecciones de [`pasos/`](../pasos) son programas completos: los corres, lees lo que imprimen y entiendes cada resultado. En los ejercicios cambias de papel. Tú escribes las partes que importan, y así comprendes la herramienta porque la usas con tus manos. Cuando un ejercicio usa un atajo de `taller_hpo`, te dice qué hace por dentro. Los primeros ejercicios usan la herramienta "a mano" (Optuna, la biblioteca que busca los ajustes, y cálculos sencillos), y los siguientes presentan el atajo que hace lo mismo.

## Cómo trabajar con cada ejercicio

1. Lee el `README.md` de la carpeta del ejercicio. Te cuenta la idea nueva, los pasos que vas a hacer y una pista por paso.
2. Abre `ejercicio.py` y busca los comentarios `# TU TURNO`. Cada uno marca un hueco.
3. Completa los huecos en orden. Mientras uno siga sin completar, `ejercicio.py` se detiene con un mensaje que te dice cuál falta.
4. Corre el verificador. Te dice `Bien: ...` por cada revisión que pasa y `Te falta: ...` por cada una que no, con lo que puedes hacer.
5. Cuando todo pase, corre `ejercicio.py` y contesta las preguntas de "Qué observar".
6. Si te atoras después de leer las pistas, abre `solucion.py`. Es el mismo archivo ya resuelto, con comentarios que explican el porqué de cada decisión.

## La escalera de 13 ejercicios

Cada ejercicio trae una sola idea nueva y se apoya en los anteriores, así que conviene hacerlos en orden. Los ejercicios se suman al repo uno por uno: si un enlace todavía no abre, ese ejercicio aún no está.

### Nivel 1: conoce tu modelo, todavía sin buscador

| Ejercicio | Qué aprendes | Lección de `pasos/` que lo amplía |
|---|---|---|
| [01 Primer entrenamiento](01_primer_entrenamiento/README.md) | Entrenar una vez y leer las métricas que devuelve. | [00](../pasos/00_linea_base.py) |
| [02 El azar importa](02_el_azar_importa/README.md) | Entrenar la misma configuración con varias semillas y medir cuánto cambia el resultado. | [00](../pasos/00_linea_base.py) |
| [03 La validación se equivoca](03_la_validacion_se_equivoca/README.md) | Comparar la exactitud de validación con la verdadera, y entender por qué tu tesis no puede medir esa verdad. | [00](../pasos/00_linea_base.py) |

### Nivel 2: tu primer buscador

| Ejercicio | Qué aprendes | Lección de `pasos/` que lo amplía |
|---|---|---|
| [04 Primera búsqueda](04_primera_busqueda/README.md) | Escribir una búsqueda con Optuna sobre una sola perilla en escala logarítmica. | [02](../pasos/02_primer_estudio.py) |
| [05 Varias perillas](05_varias_perillas/README.md) | Buscar sobre perillas de distinto tipo y comparar la mejor contra tu configuración por defecto. | [02](../pasos/02_primer_estudio.py) |
| [06 Guarda y retoma](06_guarda_y_retoma/README.md) | Guardar la búsqueda en un archivo y retomarla sin repetir intentos. | [02](../pasos/02_primer_estudio.py) |
| [07 Empieza desde tu configuración](07_empieza_desde_tu_configuracion/README.md) | Hacer que el primer intento sea tu propia configuración, y usar el atajo que arma la función objetivo. | [02](../pasos/02_primer_estudio.py) |

### Nivel 3: honestidad

| Ejercicio | Qué aprendes | Lección de `pasos/` que lo amplía |
|---|---|---|
| [08 Compara con justicia](08_compara_con_justicia/README.md) | Comparar dos configuraciones con las mismas semillas y leer el veredicto con su intervalo. | [06](../pasos/06_honestidad.py) |
| [09 Cuántas semillas](09_cuantas_semillas/README.md) | Calcular cuántas semillas necesitas para distinguir una mejora real del ruido. | [06](../pasos/06_honestidad.py) |
| [10 La ganadora inflada](10_la_ganadora_inflada/README.md) | Volver a entrenar a la ganadora con semillas nuevas y medir cuánto se había inflado. | [06](../pasos/06_honestidad.py) |

### Nivel 4: búsquedas grandes

| Ejercicio | Qué aprendes | Lección de `pasos/` que lo amplía |
|---|---|---|
| [11 Corta a tiempo](11_corta_a_tiempo/README.md) | Podar los intentos que van mal y contar cuántas épocas ahorras. | [03](../pasos/03_poda.py) |
| [12 Mejor pero más caro](12_mejor_pero_mas_caro/README.md) | Buscar con dos metas a la vez, exactitud y tamaño, sin sumarlas. | [04](../pasos/04_pareto.py) |
| [13 Conecta tu modelo](13_conecta_tu_modelo/README.md) | Adaptar un `train.py` de tesista y conectarlo a una búsqueda real. | [07](../pasos/07_conecta_tu_modelo.py) |

## Para empezar

Necesitas una tarjeta de video NVIDIA con CUDA (el sistema de NVIDIA para calcular en la tarjeta) y [uv](https://docs.astral.sh/uv/). Todos los ejercicios entrenan en la tarjeta de video. Si el taller no ve una, se detiene con un mensaje que dice qué falta.

Desde la raíz del repo, instala el entorno una sola vez:

```bash
uv sync --locked
```

En Windows con WSL2, corre esto en cada terminal nueva antes de cualquier comando de `uv`. El comando deja el entorno en el disco de Linux, donde corre más rápido:

```bash
source scripts/wsl/entorno.sh
```

Después abre el primer ejercicio, completa `ejercicio.py` y verifica tu trabajo:

```bash
uv run python ejercicios/verificar.py 01
```

El número cambia con cada ejercicio (`04` para el cuarto). Con `--solucion` el verificador revisa `solucion.py` en vez de tu archivo, para que veas qué aspecto tiene un ejercicio terminado:

```bash
uv run python ejercicios/verificar.py 01 --solucion
```

## Qué trae cada carpeta

| Archivo | Para qué sirve |
|---|---|
| `README.md` | Cuenta la idea nueva del ejercicio, los pasos y una pista por paso. |
| `ejercicio.py` | El archivo que completas. Corre como programa y, completo, imprime lo que debes observar. |
| `solucion.py` | El mismo archivo, resuelto y comentado. |
| `comprobar.py` | Las revisiones que usa el verificador. Léelo si quieres saber exactamente qué espera de tu código. |

Junto a las carpetas hay dos archivos de apoyo: `verificar.py`, que corre las revisiones, y `_comun.py`, que guarda lo que comparten.
