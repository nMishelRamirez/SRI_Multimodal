Si utilizas este corpus / conjunto de datos:

Por favor, cita: M. Hodosh, P. Young y J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artificial Intelligence Research, Volumen 47, páginas 853-899
http://www.jair.org/papers/paper3994.html

Leyendas, Divisiones del Conjunto de Datos y Anotaciones Humanas:

Flickr8k.token.txt: las leyendas originales del conjunto de datos Flickr8k. La primera columna es el ID de la leyenda, que tiene el formato "dirección de la imagen # número de la leyenda".

Flickr8k.lemma.txt: la versión lematizada de las leyendas anteriores.

Flickr_8k.trainImages.txt: Las imágenes de entrenamiento utilizadas en nuestros experimentos.

Flickr_8k.devImages.txt: Las imágenes de desarrollo/validación utilizadas en nuestros experimentos.

Flickr_8k.testImages.txt: Las imágenes de prueba utilizadas en nuestros experimentos.

ExpertAnnotations.txt: las valoraciones de los expertos. Las dos primeras columnas son los IDs de la imagen y la leyenda. Los IDs de las leyendas tienen el formato <nombre del archivo de imagen>#<0-4>. Las tres columnas siguientes son las valoraciones de los expertos para ese par imagen-leyenda. Las puntuaciones varían de 1 a 4, donde:

Un 1 indica que la leyenda no describe en absoluto la imagen.

Un 2 indica que la leyenda describe algunos aspectos menores de la imagen, pero no la describe en su totalidad.

Un 3 indica que la leyenda casi describe la imagen, con algunos errores menores.

Un 4 indica que la leyenda describe la imagen.

CrowdFlowerAnnotations.txt: las valoraciones de CrowdFlower. Las dos primeras columnas son los IDs de la imagen y la leyenda. La tercera columna es el porcentaje de "Síes", la cuarta columna es el número total de "Síes" y la quinta columna es el número total de "Noes". Un "Sí" significa que la leyenda describe la imagen (posiblemente con algunos errores menores), mientras que un "No" significa que la leyenda no describe la imagen. Cada par imagen-leyenda tiene un mínimo de tres valoraciones, pero algunos pueden tener más.