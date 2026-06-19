# Análisis estadístico VNS–IVNS

Este directorio contiene los archivos utilizados para limpiar, validar y analizar los resultados experimentales de VNS e IVNS.

Los CSV originales se leen desde `results/`. Los archivos originales no se modifican. Las tablas, los datos limpios y las figuras se generan en `results/analysis_final/`.

## Requisitos

Se recomienda utilizar Python 3.10 o una versión posterior.

Las bibliotecas necesarias son:

* NumPy
* pandas
* Matplotlib
* SciPy

Se pueden instalar mediante:

```bash
python3 -m pip install numpy pandas matplotlib scipy
```

Cuando se utiliza un entorno virtual existente, este debe activarse antes de ejecutar los análisis.

Por ejemplo:

```bash
source tesisEnv/bin/activate
```

La ubicación del entorno virtual puede variar según la instalación.

Las versiones instaladas se pueden comprobar con:

```bash
python3 - <<'PY'
import numpy
import pandas
import matplotlib
import scipy

print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("SciPy:", scipy.__version__)
PY
```

## Archivos de análisis

### `analysis_common.py`

Contiene funciones compartidas para leer datos, calcular estadísticas y localizar las carpetas del proyecto.

### `01_limpieza_validacion_total.py`

Lee todos los CSV originales, normaliza sus columnas y valida:

* cantidad de corridas;
* cantidad de trazas;
* semillas;
* algoritmos;
* instancias;
* factibilidad;
* estado de las ejecuciones;
* ausencia de duplicados;
* correspondencia entre VNS e IVNS;
* igualdad de las soluciones iniciales en los experimentos pareados.

Los datos limpios se guardan en:

```text
results/analysis_final/clean/
```

### `02_descriptivo_oficial.py`

Genera los resúmenes descriptivos y las figuras principales de los experimentos oficiales.

### `03_variantes_ivns_oficial.py`

Compara las nueve variantes oficiales de IVNS.

Se aplican:

* prueba de Kruskal-Wallis;
* comparaciones por pares mediante Mann-Whitney U;
* corrección de Holm;
* tamaño de efecto.

### `04_comparacion_oficial_vns_ivns.py`

Analiza la primera comparación oficial entre VNS e IVNS.

Se utiliza la prueba de Wilcoxon porque las corridas comparten semilla y solución inicial.

### `05_sensibilidad_tiempo_a.py`

Analiza las variantes de IVNS en las instancias de la familia A para los tiempos:

```text
0.20, 0.50, 1.00 y 1.50
```

Este análisis estudia el efecto del tiempo disponible y selecciona la configuración utilizada en el experimento confirmatorio.

### `06_confirmatorio_a.py`

Compara de forma pareada:

```text
VNS
IVNS baseline-first con tiempo 1.00
```

Se utilizan las instancias:

```text
A-n33-k5
A-n65-k9
A-n80-k10
```

Cada algoritmo recibe la misma solución inicial para cada instancia y semilla.

### `07_exploracion_f.py`

Resume los experimentos exploratorios realizados sobre:

```text
F-n135-k7
```

Incluye:

* perfiles baseline y efg-medium con tiempo 0.05;
* perfil efg-high con distintos tiempos;
* selección de la configuración efg-high-first con tiempo 1.50.

### `08_confirmatorio_f.py`

Compara de forma pareada:

```text
VNS
IVNS efg-high-first con tiempo 1.50
```

La comparación se realiza sobre `F-n135-k7`, utilizando la misma solución inicial para ambos algoritmos.

## Ejecución completa

Todos los análisis se ejecutan mediante:

```bash
cd "$HOME/Proyectos Uni/vrp"

bash analysis/run_all_analysis.sh
```

El proceso se detiene cuando alguno de los archivos presenta un error.

## Orden de ejecución

El archivo `run_all_analysis.sh` ejecuta:

```text
01_limpieza_validacion_total.py
02_descriptivo_oficial.py
03_variantes_ivns_oficial.py
04_comparacion_oficial_vns_ivns.py
05_sensibilidad_tiempo_a.py
06_confirmatorio_a.py
07_exploracion_f.py
08_confirmatorio_f.py
```

## Resultados generados

### Datos limpios

```text
results/analysis_final/clean/
```

Contiene los datos normalizados y las comprobaciones de pareo.

### Tablas

```text
results/analysis_final/tables/
```

Contiene:

* resúmenes descriptivos;
* rankings;
* diferencias entre algoritmos;
* pruebas estadísticas;
* tamaños de efecto;
* comprobaciones de validación.

### Figuras

```text
results/analysis_final/figures/
```

Las figuras se organizan en:

```text
official/
exploratory_a/
confirmatory_a/
exploratory_f/
confirmatory_f/
```

### Logs

```text
logs/estadistica/
```

Cada archivo Python genera un log independiente.

## Reconstrucción del análisis

Para volver a generar todas las tablas y figuras se elimina únicamente la carpeta de resultados procesados:

```bash
cd "$HOME/Proyectos Uni/vrp"

rm -rf results/analysis_final
rm -rf logs/estadistica

mkdir -p logs/estadistica

bash analysis/run_all_analysis.sh
```

No deben eliminarse las carpetas:

```text
results/oficiales-ivns-variants/
results/oficiales-final/
results/pruebas-personales/
results/confirmatorios/
```

Estas carpetas contienen los CSV originales de las corridas.

## Comprobación final

El reporte general se encuentra en:

```text
results/analysis_final/tables/validation_report.csv
```

Puede consultarse con:

```bash
python3 - <<'PY'
import pandas as pd

path = "results/analysis_final/tables/validation_report.csv"
report = pd.read_csv(path)

print(report.to_string(index=False))
print()
print("Estados:", report["status"].value_counts().to_dict())
PY
```

Todos los controles deben aparecer con el estado:

```text
ok
```
