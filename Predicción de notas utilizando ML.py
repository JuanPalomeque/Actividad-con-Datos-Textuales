#!/usr/bin/env python
# coding: utf-8

# ## PREDICCION DE NOTAS UTILIZANDO APRENDISAJE AUTOMATICO

# In[550]:


from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import textacy.preprocessing as tprep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


# ### Cargo la base de datos

# In[551]:


df = pd.read_csv("IOrespuestas.csv")
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings('ignore')


# ### Preparo el Corpus para el trabajo

# In[552]:


# SELLECIONO LAS COLUMNAS,CAMBIO EL NOMBRE, ELIMINO LAS NA Y TRABAJO EL FORMATO

df_corpus=df[['Padrón sin números', '¿qué es la Investigación Operativa?']]
df_corpus.columns = ['padrón', 'texto']
df_corpus['notas correctas'] = pd.read_csv(r"Resultado_Notas_queEsIO_Correcto.csv")
df_corpus =df_corpus.dropna()
df_corpus['texto'] = df_corpus['texto'].astype('U')


# ### Preprocesamiento

# In[553]:


wordnet_lematizer = WordNetLemmatizer()
spanish_stemmer = SnowballStemmer('spanish')

def preprocesar(texto):
    '''función que preprocesa el texto para ser trabajado'''
    
    tokens = [ tprep.remove.accents(str(w).lower()) for w in word_tokenize(texto) if w.isalpha() if len(w)>4]
    tokens = [ spanish_stemmer.stem(t) for t in tokens ]
    return tokens


# ### Empiezo a entrenar el modelo

# In[570]:


# DEFINO LOS DATOS DE ENTRENAMIENTO Y PRUEBA (80%/20%)
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(df_corpus['texto'], df_corpus['notas correctas'], test_size=0.2) 


# In[571]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(tokenizer = preprocesar, stop_words=stopwords.words('spanish'))


# In[572]:


#INSTANCIO LAS TRANSFORMADAS DE LA VARIABLES EXPLICATIVAS
count_train = cv.fit_transform(x_train)
count_test = cv.transform(x_test)


# In[573]:


#USO DEL MODELO DE NAIVE-BAYES

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

count_nb = MultinomialNB()
count_nb.fit(count_train, y_train)
count_nb_pred = count_nb.predict(count_test)
count_m = metrics.accuracy_score(y_test,count_nb_pred)


# In[574]:


from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(tokenizer = preprocesar, stop_words=stopwords.words('spanish') )
tfidf_train = tv.fit_transform(x_train.values.astype('U'))
tfidf_test = tv.transform(x_test.values.astype('U'))


# In[575]:


# MIDO LA PRECISION DEL MODELO
f'La precisión de la predicción respecto a los datos de entrenamiento es del {round(count_m*100,2)}%' 


# In[576]:


#Realizo una prueba sobre un texto de ejemplo

texto_de_prueba = "Es la optimización mediante el modelado y aplicación de la matemática"

texto_de_prueba_vectorizado_cv = cv.transform([texto_de_prueba])
t_count_pred = count_nb.predict(texto_de_prueba_vectorizado_cv)
f'se predice la nota de "{int(t_count_pred) }" para el texto de prueba'


# ### Defino la función evaluadora

# In[577]:


def asignar_notas_ML(texto):
    '''función que asigna el modelo de Naive-Bayes usando texto y notas como variables'''
    notas_ML = [count_nb.predict(cv.transform([t])) for t in [texto] ]
    return int(sum(notas_ML))


# ### Se aplica la función evaluadora y se obtienen los resultados

# In[578]:


df_corpus['notas_ML'] = df_corpus['texto'].apply(asignar_notas_ML)


# In[579]:


df_corpus.head()


# ## ANALISIS DE LOS RESULTADOS

# In[580]:


df_analisis = df_corpus.copy()


# In[581]:


df_analisis['notas test'] = pd.read_csv(r"C:\Users\User\Ejercicios de Pyhton con Anaconda\Pre Ingenieria\Resultado_Notas_queEsIO_testPrueba.csv")


# In[582]:


df_analisis['bias modelo'] = abs(df_corpus['notas correctas']-df_corpus['notas_ML'])
# Porcentaje de respuestas correctas modelo
delta_ML = round((1 - ( df_analisis[df_analisis['bias modelo'] != 0]['notas_ML'].count() / len(df_analisis) ) )*100,2)
print(f' Con el modelo se ubtuvo un {delta_ML} % de respuestas correctas')


# In[583]:


df_analisis['bias manual'] = abs(df_corpus['notas correctas']-df_analisis['notas test'])
# Porcentaje de respuestas correctas de las respuestas de control
delta_manual = round((1 - ( df_analisis[df_analisis['bias manual'] != 0]['notas test'].count() / len(df_analisis) ) )*100,2)
print(f' Corrigiendo a mano se obtuvo un {delta_manual} % de respuestas correctas')


# In[584]:


df_analisis.head()


# In[585]:


names = ['bias de la evaluación manual', 'bias de la evaluación del modelo']
values = [delta_ML, delta_manual]

plt.figure(figsize=(20, 5))

plt.subplot(131)
plt.bar(names, values)
plt.ylim([0, 100])
plt.title('Bias de los resultados')
plt.show()


# In[538]:


# Presición del modelo

#Porcentaje de coincidencias en las respuestas del modelo con las respustas correctas (en 10 corridas del modelo)
Coincidencias = [89.92, 87.39, 85.71, 87.39,87.39,85.71,88.24, 83.19,84.87,85.71]
precisión_promedio = sum(Coincidencias)/len(Coincidencias)
f'La precisión promedio del modelo es de {round(precisión_promedio,2)} %'


# In[586]:


# Varianza del modelo

from numpy import var
f' Con las 10 corridas del modelo se obtuvo una variación de {round(var(Coincidencias),2)} % '


# In[587]:


df_corpus.to_excel(r"Resultado_Notas_queEsIO_ML.xlsx")


# In[588]:


df_analisis.to_excel(r"Analisis_Notas_queEsIO.xlsx")


# In[ ]:




