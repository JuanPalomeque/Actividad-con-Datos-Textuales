#!/usr/bin/env python
# coding: utf-8

# # Actividad con Datos Textuales

# In[3]:


from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd

Cargo la base de datos
# In[4]:


df = pd.read_csv("C:/Users/User/Documents/Facu/9216/IOrespuestas.csv")
pd.options.mode.chained_assignment = None  # default='warn'


# Comienzo a preparar el data frame para el trabajo

# In[170]:


# SELLECIONO LAS COLUMNS QUE QUIERO Y LES CAMBIO EL NOMBRE

df2=df[['Padrón sin números', '¿qué es la Investigación Operativa?']]
df2.columns = ['padrón', 'texto']
df2


# # ------------  Defino la base de datos  -------------------------------------------

# In[171]:


df2['notas'] = pd.read_csv("C:/Users/User/Desktop/Notas de algoritmo.csv")


# In[172]:


df2


# In[173]:


df3 = df2.drop(columns = 'padrón')
df3


# In[192]:


x_train, x_test,y_train,y_test = train_test_split(df3['texto'], df3['notas'], test_size=0.20, random_state=53)


# In[193]:


cv = CountVectorizer(stop_words=stopwords.words('spanish'))


# In[194]:


count_train = cv.fit_transform(x_train.values.astype('U'))


# In[195]:


count_test = cv.transform(x_test.values.astype('U'))


# In[196]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# In[197]:


nb = MultinomialNB()
nb.fit(count_train, y_train)
pred = nb.predict(count_test)
metrics.accuracy_score(y_test, pred)


# In[133]:


metrics.confusion_matrix(y_test, pred, labels=[0,1,2])


# In[138]:


dict = {'respuestas': x_test, 'predicción':pred}


# In[139]:


df_res = pd.DataFrame(dict)


# In[144]:


df_res


# ## Análisis
# 

# # Cálculo del mejor alpha

# In[204]:


alphas = np.arange(0,1,0.05)
def calcular_alpha(alphas):
     nb = MultinomialNB(alpha=alpha)
     nb.fit(count_train, y_train)
     pred = nb.predict(count_test)
     return metrics.accuracy_score(y_test, pred)


# In[205]:


valor=[]
i=0
for alpha in alphas:
    valor.append(calcular_alpha(alpha))
    i+=1
    


# In[206]:


valor = np.array(valor)
valor


# In[207]:


plt.plot(alphas,valor)
plt.show()


# In[208]:


df_res.to_excel("C:/Users/User/Desktop/resultados.xlsx")


# In[ ]:




