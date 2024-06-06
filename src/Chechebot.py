from discord.ext.commands.core import check
import nltk #Lenguaje Natural
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
from tensorflow.python.framework import ops
#from tensorflow.python.util.nest import is_sequence
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
#from tensorflow.python.util.nest import is_sequence_or_composite
import json
import random
import pickle
import discord
import os

import pandas as pd
from math import sqrt
from metodos import test
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from urllib import parse, request
import re
import youtube_dl
import wikipedia
from discord.ext import commands
#nltk.download('punkt')#Descomentar si algo falla

#Variables globales
archivoJson="./data/contenido.json"
rutaPickle="./data/variables.pickle"
modeloTflearn="./data/modelo.tflearn"

llave = ""#Remplazar por llave de su servidor
neuronas = 10
ver=2500 #veces que se vera para clasificar
numPalabras=10 #numero de palabras en el patron
palabras=[]
tags=[]
auxX=[]
auxY=[]
#cargamos los datos de las peliculas
movies = pd.read_csv('./data/games.csv')
#Cargamos los datos de los usuarios 
ratings = pd.read_csv('./data/ratingsG.csv')
#Entrenamiento
with open(archivoJson, encoding='utf-8') as archivo:
        datos = json.load(archivo)

def interfaz():
    #print(os.listdir('.'))
    print("\nHola Bienvenido al mejor bot de la zona:\n",
    "\nPresiona 1 para actualizar o crear datos del modelo.",
    "\nPresiona 2 Para ejecutar el Chechebot localmente.",
    "\nPresiona 3 para ejecutar el Chechebot en test-mode.",
    "\nPresiona 4 para lanzar el Chechebot en Discord.",
    "\nPresiona 5 para ejecutar el Chechebot Recargado con todas las funcionalidades",
    "\nPresiona 6 para Salir.")


def actualizar():
    print("Actualizando")
    
    global palabras
    global tags
    global auxX
    global auxY
    global modelo
    #palabras=[]
    #tags=[]
    #auxX=[]
    #auxY=[]

    #Acomodamos las palabras de acuerdo a sus Tags
    for contenido in datos["contenido"]:#Recorremos para ir guardando las palabras y tags
        for patrones in contenido["patrones"]:
            auxPalabra =nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])
            if contenido["tag"] not in tags:#Para no repetir tags
                tags.append(contenido["tag"])

    #Ordenamos las palabras y Tags
    palabras = [stemmer.stem(w.lower()) for w in palabras if w!='?']
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida = []

    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra=[stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)#Asignamos valores de 1 y 0 si se encuentra o no la palabra
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[tags.index(auxY[x])]=1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = numpy.array(entrenamiento)#Lo hacemos array
    salida = numpy.array(salida)#Lo hacemos Array

    with open(rutaPickle, "wb") as archivoPickle:
        pickle.dump((palabras,tags,entrenamiento,salida),archivoPickle)#Guardamos datos
        
    ops.reset_default_graph()
    #Creacion de la red neuronal
    red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")#La salida a base de proba
    red = tflearn.regression(red)

    modelo = tflearn.DNN(red,checkpoint_path=None,session=None)
    #Para ejecutar eel entrenamiento, con ver se usa el numero de repeticiones
    modelo.fit(entrenamiento,salida,n_epoch=ver,batch_size=numPalabras,show_metric=True)
    modelo.save(modeloTflearn)#Guardamos el modelo

def Chechebot():
    #Cargamos datos
    with open(rutaPickle,"rb") as archivoPickle:
        palabras,tags,entrenamiento,salida = pickle.load(archivoPickle)
    ops.reset_default_graph()

    red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")#La salida a base de proba
    red = tflearn.regression(red)

    modelo = tflearn.DNN(red,checkpoint_path=None,session=None)
    modelo.load(modeloTflearn)

    print("Ejecutando localmente")
    while True:
        entrada = input("Tu: ")
            
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)
        entradaProcesada = [stemmer.stem(palabrra.lower()) for palabrra in entradaProcesada]
        for palabraIndividual in entradaProcesada: #Hot spots (se hace lo del cafe pues)
            for i,palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])#Se hace la prediccion de que tema habla
        resultadosIndices = numpy.argmax(resultados)#Tomamos el tag maximo
        tag = tags[resultadosIndices] #Obtenemos el Tag, con valor maximo obtenido.

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]#Obtenemos el array correspondiente
        print("ChecheBot: ",random.choice(respuesta))#Seleccionamos una opcion random
        if(entrada == "Adios"):
            return

def ChechebotTestMode():
    with open(rutaPickle,"rb") as archivoPickle:
        palabras,tags,entrenamiento,salida = pickle.load(archivoPickle)
    ops.reset_default_graph()

    red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")#La salida a base de proba
    red = tflearn.regression(red)

    modelo = tflearn.DNN(red,checkpoint_path=None,session=None)
    modelo.load(modeloTflearn)
    os.system("clear")
    print("Hora de Testear")
    while True:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada)
        entradaProcesada = [stemmer.stem(palabrra.lower()) for palabrra in entradaProcesada]
        for palabraIndividual in entradaProcesada:
            for i,palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]
        
        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]
        auxTag= []
        e=0
        for i in resultados[0]:
            cadena = str(str(tags[e])+" : "+ str(i))
            auxTag.append(cadena)
            e = e+1
        
        print(auxTag)
        print("ChecheBot: ",random.choice(respuesta))
        if(entrada == "Adios".lower()):
            return

def ChecheBotDiscord():
    #Cargamos datos
    global llave
    with open(rutaPickle,"rb") as archivoPickle:
        palabras,tags,entrenamiento,salida = pickle.load(archivoPickle)
    ops.reset_default_graph()

    red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")#La salida a base de proba
    red = tflearn.regression(red)

    modelo = tflearn.DNN(red,checkpoint_path=None,session=None)
    modelo.load(modeloTflearn)
    os.system('clear')
    print("Nos vamos a Discord")
    intents = discord.Intents.default()
    intents.message_content = True
    cliente = discord.Client(intents=intents)
    print("Listo Para discord")
    while True:
        #Evento
        @cliente.event
        async def on_message(mensaje):#si enviamos un meensaje, este nos contestara
            print(mensaje.content)
            if mensaje.author == cliente.user:#Para que no Hable consigo mismo
                return
            #Se hace lo mismo que arriba
            cubeta = [0 for _ in range(len(palabras))]
            entradaProcesada = nltk.word_tokenize(mensaje.content)
            entradaProcesada = [stemmer.stem(palabrra.lower()) for palabrra in entradaProcesada]
            for palabraIndividual in entradaProcesada:
                for i,palabra in enumerate(palabras):
                    if palabra == palabraIndividual:
                        cubeta[i] = 1
            resultados = modelo.predict([numpy.array(cubeta)])
            resultadosIndices = numpy.argmax(resultados)
            tag = tags[resultadosIndices]
            print("Tag: ",tag)

            for tagAux in datos["contenido"]:
                if tagAux["tag"] == tag:
                    respuesta = tagAux["respuestas"]
                    print("res: ", respuesta)
            #Aqui enviamos el mensaje a traves del canal que se le contacto 
            #Respuestas largas
            print("Longitud: ",len(random.choice(respuesta)))
            embed = discord.Embed(title="Titulo ejemplo", description=random.choice(respuesta))        
            #await mensaje.channel.send(random.choice(respuesta))
            await mensaje.channel.send(embed=embed)
        cliente.run(llave)#llave para que corra

def main():
    os.system('clear')
    while True:
        interfaz()
        x=int(input())
        if(x==1):
            actualizar()
        elif(x==2):
            Chechebot()
        elif(x==3):
            ChechebotTestMode()
        elif(x==4):
            ChecheBotDiscord()
        elif(x==5):
            ChechebotTotal()
        elif(x==6):
            break
        else:
            print("Elije una opcion valida")

def ChechebotTotal():
    global llave
    with open(rutaPickle,"rb") as archivoPickle:
        palabras,tags,entrenamiento,salida = pickle.load(archivoPickle)
    ops.reset_default_graph()

    red = tflearn.input_data(shape=[None,len(entrenamiento[0])])
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,neuronas)#conectamos
    red = tflearn.fully_connected(red,len(salida[0]),activation="softmax")#La salida a base de proba
    red = tflearn.regression(red)

    modelo = tflearn.DNN(red,checkpoint_path=None,session=None)
    modelo.load(modeloTflearn)
    os.system('clear')
    print("Nos vamos a Discord")

    bot = commands.Bot(command_prefix='>', description="The best bot in the world Baby.",intents=discord.Intents.all())
    cliente = discord.Client(intents=discord.Intents.default())

    @bot.command()
    async def ping(ctx):
        print("Pong")
        await ctx.send("pong")

    @bot.command()
    async def suma(ctx, num1: float, num2: float):
        await ctx.send(num1 + num2)

    @bot.event
    async def on_ready():
        print("My bot is ready")

    @bot.command()
    async def youtube(ctx,*,search):
        consulta = parse.urlencode({'search_query': search})
        respuesta = request.urlopen('http://www.youtube.com/results?' + consulta)
        resultados = re.findall(r'/watch\?v=(.{11})', respuesta.read().decode())
        await ctx.send('https://www.youtube.com/watch?v=' + resultados[0])
    
    @bot.command(pass_content=True)
    async def recomienda(ctx):
        #Se ponen parentesis para evitar conflictos con peliculas que pueden tener años en sus titulos
        movies['year'] = movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
        #Quitamos los parentesis
        movies['year'] = movies.year.str.extract('(\d\d\d\d)',expand=False)
        #Quitamos el año de la columna title
        movies['title'] = movies.title.str.replace('(\(\d\d\d\d\))', '')
        #quitamos todos los espacios en blanco usando strip
        movies['title'] = movies['title'].apply(lambda x: x.strip())
        gamesSurvey= test()
        await ctx.send("A continuación se te presentarán 5 juegos\nPor favor evalualos con una nota entre el 1 y el 5, puedes usar un decimal si lo deseas")
        await ctx.send(str(gamesSurvey[0]))
        def check(msg):
            return msg.author == ctx.author and msg.channel == ctx.channel and \
            msg.content.lower() in ['1','2','3','4','5','1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9','5.0']
        message_response = await bot.wait_for('message', check=check)
        juego1 = float(message_response.content)
        #Meter whiles de verificacion
        await ctx.send(str(gamesSurvey[1]))
        message_response = await bot.wait_for('message', check=check)
        juego2 = float(message_response.content)
        await ctx.send(str(gamesSurvey[2]))
        message_response = await bot.wait_for('message', check=check)
        juego3 = float(message_response.content)
        await ctx.send(str(gamesSurvey[3]))
        message_response = await bot.wait_for('message', check=check)
        juego4 = float(message_response.content)
        await ctx.send(str(gamesSurvey[4]))
        message_response = await bot.wait_for('message', check=check)
        juego5 = float(message_response.content)

        userInput = [
            {'title':gamesSurvey[0], 'rating':float(juego1)},
            {'title':gamesSurvey[1], 'rating':float(juego2)},
            {'title':gamesSurvey[2], 'rating':float(juego3)},
            {'title':gamesSurvey[3], 'rating':float(juego4)},
            {'title':gamesSurvey[4], 'rating':float(juego5)}
         ] 
        inputMovies = pd.DataFrame(userInput)
        #Filtramos las peliculas por titulo
        inputId = movies[movies['title'].isin(inputMovies['title'].tolist())]
        #ordenamos las peliculas por id
        inputMovies = pd.merge(inputId, inputMovies)
        #Quitamos el año
        inputMovies = inputMovies.drop('year',axis=1)

        #Buscamos los usuarios que tambien "vieron" la pelicula
        userSubset = ratings[ratings['movieId'].isin(inputMovies['movieId'].tolist())]
        #quitar comentario para ver tabla
        #userSubset.head()
        userSubsetGroup = userSubset.groupby(['userId'])
        #Ordenmos por usuarios con la pelicula mas vista 
        userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
        userSubsetGroup[0:3]
        userSubsetGroup = userSubsetGroup[0:100]
        #Usaremos la coorrelacion de pearson, la guardamos en un diccionario, la llave es el id 
        #del usuario.
        pearsonCorrelationDict = {}
        #Para cada grupo de usuarios 
        for name, group in userSubsetGroup:
            #Ordenamos la entrada y el grupo de usuarios para que no se mezclen
            group = group.sort_values(by='movieId')
            inputMovies = inputMovies.sort_values(by='movieId')
            #Obtenemos N para la formula
            nRatings = len(group)
            #
            #Obtenemos Las reseñas de los usuarios con peliculas en comun
            temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
            tempRatingList = temp_df['rating'].tolist()
            #Ponemos las reseñas del grupo en una lista
            tempGroupList = group['rating'].tolist()
            #Calculamos la correlacion de pearson entre los usuarios (x & y)
            Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
            Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
            Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
            
        #Si el denomindor es 0 entonces la correlación es 0 
            if Sxx != 0 and Syy != 0:
                pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
            else:
                pearsonCorrelationDict[name] = 0

        #obtener las películas vistas por los usuarios en nuestro pearsonDF desde el dataframe 
        #de calificaciones y luego almacenar su correlación en una nueva columna _similarityIndex ".
        pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
        print("DF: ",pearsonDF)
        pearsonDF.columns = ['similarityIndex']
        pearsonDF['userId'] = pearsonDF.index
        pearsonDF.index = range(len(pearsonDF))
        #Obtenemos los 50 usuarios que se parecen mas 
        topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
        topUsersRating=topUsers.merge(ratings, left_on='userId', right_on='userId', how='inner')
        #Multiplicamos la similitud por las calificaciones del usuario
        topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
        # Aplicamos una suma de los usuarios principales después de agruparlos por ID 
        tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
        tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']

        #Creamos un dataframe vacio
        recommendation_df = pd.DataFrame()
        #Tomamos media ponderada
        recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
        recommendation_df['movieId'] = tempTopUsersRating.index
        #Obtenemos las mejores peliculas recomendadas y las ordenamos por id
        recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
        #Quitar comentario para ver tabla
        await ctx.send("estamos buscando recomendaciones para ti, espera un segundo")
        for _ in range(5):
            time.sleep(1)
            await ctx.send("*")
        await ctx.send("Basado en usuarios que dieron calificaciones similares a las tuyas\nTe recomendamos estos juegos")
        await ctx.send(str(movies.loc[movies['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())].drop('movieId', 1)))
    
    @bot.event
    async def on_message(mensaje):
        #print(type(mensaje.content))
        if mensaje.author == bot.user or mensaje.content[0] == '>' or mensaje.content in ['1','2','3','4','5','1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9','5.0']:#Para que no Hable consigo mismo
            await bot.process_commands(mensaje)
            return
        #Se hace lo mismo que arriba
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(mensaje.content)
        entradaProcesada = [stemmer.stem(palabrra.lower()) for palabrra in entradaProcesada]
        for palabraIndividual in entradaProcesada:
            for i,palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1
        resultados = modelo.predict([numpy.array(cubeta)])
        resultadosIndices = numpy.argmax(resultados)
        tag = tags[resultadosIndices]

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuesta = tagAux["respuestas"]
                print("Longitud: ",len(random.choice(respuesta)))
        if(len(random.choice(respuesta) >= 2000) and len(random.choice(respuesta) <= 6000)):
            embed = discord.Embed(title="Receta", description=random.choice(respuesta))        
            await mensaje.channel.send(embed=embed)
        else:       
            await mensaje.channel.send(random.choice(respuesta))

        # INCLUDES THE COMMANDS FOR THE BOT. WITHOUT THIS LINE, YOU CANNOT TRIGGER YOUR COMMANDS.
        await bot.process_commands(mensaje)
    
    bot.run(llave)

main()
