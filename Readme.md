# Estructura del Repositorio
El repositorio sigue la siguient estructura:
.
|
|_ data
|_ docs
|_ notebooks
|_ src

Donde:
- data: Posee las hojas de datos, archivos con información para procesar, asi como salidas de los programas una vez procesados.
- docs: Se encuentra el PDF de lo que se hizo en el proyecto asi como el objetivo de este.
- notebooks: Se encuentra los notebooks usados, en este caso se encuentra el notebook del Recomendador.
- Se encuentran los archivos fuente del proyecto, son un conjunto de archivos `.py`

# Objetivo
En el pdf se encuentra más desarrollado, pero el objetivo principal del proyecto es presentar dos sistemas, uno de recomendación y otro un chatbot que proporciona recetas de cocina.

# Requisitos para el sistema
Para que el sistema se totalmente fluido deben instalar los siquiente paquetes:
1.`pip install -U discord.py`
2.`pip install tflearn`
3.`pip install nltk`
4.`pip install discord.py`
5.`pip install numpy`
6.`pip install tensorflow`
7.`pip install Pillow==9.5.0`
8.`pip install youtube_dl`

Asi mismo usar la versión de `python 3.8-3.10`, de otra manera es posible que alguna parte del codigo no llegue a funcionar.

Aunque si bien la mayoria del proyecto se puede ejeuctar sin discord necesariamente, los pasos a seguir son:
1. Comentar los import de Discord
2. Comentar las funciones `ChecheBotDiscord()` y `ChecheBotTotal()`asi como donde hagan sus apariciones.
3. Una vez hecho esto podremos lanzar el chatbot en un ambiente local asi como de pruebas.
4. Para la parte del recomendador, se agrega un notebook que sigue todos los pasos con algunos comentarios.

# Ejecutar
Para ejecutar nos posicionamos en la raíz y ejecutamos `python src/Chechebot.py` de esta manera se ejecutara.
En caso de que se ejecute a nivel del archivo `.py` deberan actualizar las rutas.

# Opcional
En caso de encontrar conflictos, es posible que lo recomendable sea crear un entorno virtual ya sea con Python o con Conda.
El sistema de Chatbot no funciona en computadoras con procesadores M de Apple, pues al ejecutar indica que no encuentra la instrucción solicitada, por lo que es ampliamente recomendable que se ejecute en un Procesador Intel o AMD (evitar arquitecturas ARM)

En caso de querer crear el bot y lanzarlo a discord, en este enlace https://dev.to/alexanderg/como-crear-un-bot-en-discord-py-4hgc se encuentra los pasos mas o menos detallados. Finalmente, cuando lo generen en su servidor, deberan poner la "llave" en el atributo llave, este tambien es conocido como Token.