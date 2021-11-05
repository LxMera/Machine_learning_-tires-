from google.colab import files
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pickle
import pandas as pd
from IPython.display import clear_output

df=pd.DataFrame(columns=('Name','SVM','SVM_Kernel_linear','MLP_3_Layers','MLP_2_Layers','Random_Forest', 'Class'))

def get_models():
  print('Cargando modelos...')
  os.system('wget https://www.dropbox.com/s/f62r2wkdvsy33bh/models.zip?dl=1')
  os.system('mv models.zip?dl=1 models.zip')
  os.system('unzip models.zip')

  model1 = pickle.load(open('/content/content/SVM.sav', 'rb'))
  model2 = pickle.load(open('/content/content/SVM_Kernel_linear.sav', 'rb'))
  model3 = pickle.load(open('/content/content/MLP_3layers.sav', 'rb'))
  model4 = pickle.load(open('/content/content/MLP_2layers.sav', 'rb'))
  model5 = pickle.load(open('/content/content/RandomForest.sav', 'rb'))

  print('...hecho')
  return model1, model2, model3, model4, model5

def clasificar(model):
  Class=['flat', 'full']
  name=files.upload()
  ima=cv2.imread(next(iter(name)))
  ima=((ima-np.min(ima))/(np.max(ima)-np.min(ima))).astype('float32')
  
  sz=np.shape(ima)
  flat=ima[...,0].reshape(1,sz[0]*sz[1])

  #Matriz de datos de los algoritmos combinados
  x1=model[0].predict_proba(flat)[:,0].reshape((np.shape(flat)[0],1))
  x2=model[1].predict_proba(flat)[:,0].reshape((np.shape(flat)[0],1))
  x3=model[2].predict_proba(flat)[:,0].reshape((np.shape(flat)[0],1))
  x4=model[3].predict_proba(flat)[:,0].reshape((np.shape(flat)[0],1))
  x_data=np.concatenate((x1,x2,x3,x4), axis=1)
  x1=model[4].predict_proba(x_data)[0]

  plt.figure(figsize=(6,6))
  plt.imshow(ima)
  plt.title(Class[np.argmax(x1)])
  plt.axis('off')
  plt.show()

  if not os.path.exists('results.csv'):
    df.to_csv('results.csv')
  df2=pd.read_csv('results.csv')
  df2=df2.append({'Name': next(iter(name)),
                  'SVM': x_data[0][0],
                  'SVM_Kernel_linear': x_data[0][1],
                  'MLP_3_Layers': x_data[0][2],
                  'MLP_2_Layers': x_data[0][3],
                  'Random_Forest': x1[0],
                  'Class': Class[np.argmax(x1)]} , ignore_index=True)
  df2=df2.drop(df2.columns[:np.where(df2.columns=='Name')[0][0]], axis=1)
  df2.to_csv('results.csv')
  print('La tabla muestra la probabilidad de pertenecer a la clase desinflado (flat).')
  return df2