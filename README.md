# 🎨 Método del Codo (Elbow Method) aplicado al tratamiento de imágenes

## 📌 ¿Cuántos colores necesita realmente una imagen?

Una imagen puede contener miles de colores diferentes. Sin embargo, muchos de estos colores son muy similares entre sí.

En este proyecto utilizaremos **K-Means** y el **Método del Codo (Elbow Method)** para descubrir cuántos colores representativos necesita una imagen sin perder demasiado su apariencia visual.

La idea es sencilla:

> 🖼️ **Imagen → Píxeles (RGB) → K-Means → Elbow Method → Número óptimo de colores → Imagen reducida**

---
## 🌐 Simulador Web en Vivo

Se desplego una aplicación web interactiva donde puedes dibujar tus propios datos y ver cómo funciona el algoritmo de K-medias paso a paso.

👉 <a href="https://josarta.github.io/Elbow_Method/" target="_blank" rel="noopener noreferrer"><strong>Entrar al K-Means Playground en Vivo</strong></a>

---

## 🧪 2. Análisis con Python y Scikit-Learn
Para complementar la experiencia visual,  dispongo  un cuaderno de Jupyter que toma los datos que exportas de la web y los analiza utilizando librerías científicas de Python.


### ¿Cómo ver y ejecutar el análisis?

* <strong>Vista Rápida (Estática):</strong> Puedes ver el código y las gráficas generadas haciendo clic directamente en: <a href="notebooks/ElbowMethod.ipynb" target="_blank" rel="noopener noreferrer">Ver ElbowMethod.ipynb</a>.
* <strong>Ejecutar en la Nube (Interactivo):</strong> Haz clic en el siguiente botón para abrir el código en Google Colab, donde podrás correr el modelo en tiempo real e interactuar con el simulador embebido:

<a href="https://colab.research.google.com/josarta/Elbow_Method/blob/main/notebooks/ElbowMethod.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

### ¿Cómo ver y ejecutar el análisis V1?

* <strong>Vista Rápida (Estática):</strong> Puedes ver el código y las gráficas generadas haciendo clic directamente en: <a href="notebooks/ElbowMethodV1.ipynb" target="_blank" rel="noopener noreferrer">Ver ElbowMethodV1.ipynb</a>.
* <strong>Ejecutar en la Nube (Interactivo):</strong> Haz clic en el siguiente botón para abrir el código en Google Colab, donde podrás correr el modelo en tiempo real e interactuar con el simulador embebido:

<a href="https://colab.research.google.com/josarta/Elbow_Method/blob/main/notebooks/ElbowMethodV1.ipynb" target="_blank" rel="noopener noreferrer">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

---

## 🎯 Objetivo

Determinar un número adecuado de grupos (`K`) para representar los colores principales de una imagen.

Para esto probaremos diferentes valores de `K` y analizaremos la **inercia** de cada modelo.

El objetivo no es encontrar necesariamente el menor número de colores, sino encontrar un equilibrio entre:

- 🎨 Calidad visual
- 💾 Reducción de información (compresión)
- ⚡ Complejidad del modelo
- 📊 Número de clusters

---

# 🧠 1. ¿Qué es el Método del Codo?

El **Método del Codo (Elbow Method)** es una técnica heurística utilizada para determinar un número adecuado de clusters en algoritmos de agrupamiento como **K-Means**.

La idea consiste en entrenar diferentes modelos utilizando varios valores de `K`.

Por ejemplo:

```text
K = 2
K = 3
K = 4
K = 5
K = 6
K = 7
K = 8
...
```

Para cada valor de `K`, calculamos la **Inercia** (o **Suma de Errores al Cuadrado - SSE**). La inercia mide qué tan dispersos están los puntos (en este caso, los colores de los píxeles) respecto a sus centroides asignados. Su fórmula matemática es:

\\[SSE = \sum_{j=1}^{K} \sum_{x_i \in S_j} \|x_i - \mu_j\|^2\\]

A medida que aumentamos el número de grupos (`K`), la inercia siempre disminuye (si tuviéramos tantos grupos como píxeles, la inercia sería cero). Sin embargo, llegará un punto donde añadir más grupos no aportará una mejora significativa en la reducción del error. 

Si graficamos la **Inercia vs. K**, observaremos una curva similar a un brazo humano. El punto de inflexión donde la caída del error se estabiliza (se vuelve más horizontal) se conoce como **"el codo"**, y representa el número óptimo de grupos.

---

# 🛠️ 2. Estructura del Código en Python

Para procesar la imagen, cargaremos sus píxeles en un formato tridimensional (Alto, Ancho, Canales RGB) y los transformaremos en una matriz bidimensional compatible con Scikit-Learn.

### **Paso A: Importación de Librerías y Carga de la Imagen**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

# 1. Cargar la imagen utilizando Pillow
imagen_original = Image.open("tu_imagen.jpg")
ancho, alto = imagen_original.size

# 2. Convertir la imagen a una matriz NumPy de píxeles RGB
pixeles = np.array(imagen_original)

# 3. Transformar la matriz a 2D: (Número de píxeles, 3 canales RGB)
# Cada fila será un píxel con sus valores [R, G, B] escalados entre 0 y 1
X_pixeles = pixeles.reshape(-1, 3) / 255.0
```

### **Paso B: Aplicación del Método del Codo**
Entrenamos múltiples modelos de K-Means variando \\(K\\) de 2 a 10 para registrar la inercia de cada uno:

```python
inercias = []
rango_k = range(2, 11)

for k in rango_k:
    # Usamos inicialización k-means++ para evitar mínimos locales deficientes
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_pixeles)
    inercias.append(kmeans.inertia_) # Guardamos la dispersión (SSE)

# Graficar el método del codo
plt.figure(figsize=(8, 5))
plt.plot(rango_k, inercias, marker='o', color='#cba6f7', linewidth=2)
plt.title('Método del Codo para Selección de Colores (K)', fontsize=14, fontweight='bold')
plt.xlabel('Número de Colores (K)', fontsize=12)
plt.ylabel('Inercia (SSE)', fontsize=12)
plt.xticks(rango_k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

### **Paso C: Reconstrucción de la Imagen con el K Óptimo**
Una vez identificado el "codo" en la gráfica (supongamos que \\(K=5\\)), entrenamos el modelo final y reemplazamos el color original de cada píxel por el de su centroide correspondiente:

```python
k_optimo = 5  # Reemplazar con el valor óptimo del codo observado

# Entrenar modelo final
kmeans_final = KMeans(n_clusters=k_optimo, init='k-means++', random_state=42, n_init=10)
etiquetas = kmeans_final.fit_predict(X_pixeles)
centroides_colores = kmeans_final.cluster_centers_

# Reconstruir los píxeles asignándoles su color centroide
pixeles_comprimidos = centroides_colores[etiquetas]

# Volver a dar forma tridimensional a la imagen (Alto, Ancho, RGB)
pixeles_comprimidos = (pixeles_comprimidos.reshape(alto, ancho, 3) * 255).astype(np.uint8)

# Guardar y mostrar la imagen final resultante
imagen_reducida = Image.fromarray(pixeles_comprimidos)
imagen_reducida.save("imagen_reducida.jpg")
imagen_reducida.show()
```

---

# 📊 3. Análisis de Resultados

Al ejecutar este flujo, podrás comparar visualmente cómo se reduce el peso físico de la imagen frente a la pérdida de calidad visual:

1. **Compresión:** Al pasar de millones de colores posibles a solo un pequeño puñado de grupos ($K$), reducimos drásticamente la cantidad de información necesaria para almacenar la imagen.
2. **Segmentación Semántica:** Notarás cómo el fondo, las sombras y los objetos principales se unifican en bloques continuos de color, lo cual es la base para tareas de visión por computadora como la detección de contornos.

---

*Proyecto desarrollado como parte de las prácticas de Aprendizaje No Supervisado.*

---
