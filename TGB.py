# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:44:49 2022

@author: pietro
"""

import cv2
import numpy as np
import os, sys
from matplotlib import pyplot as plt



# renomeando ordenadamente os arquivos que são adicionados na database desejada
j = 0
pathN = 'C:/Users/pietr/OneDrive/Área de Trabalho/Facul/2022-1/Processamento digital de sinais/TGB/treinamento_retinas/DATABASE_DIRET'
for filename in os.listdir(pathN):
    os.rename(os.path.join(pathN,filename), os.path.join(pathN,'diret'+str(j)+'.jpg'))
    j = j + 1


# Converter as imagens da database para GRAY; Binarizar e encontrar contorno do círculo ocular;
# Cortar a imagem nos limites do círculo; Uniformizar o tamanho;
# Aplicar Filtro Gaussiano; Aplicar CLAHE e salvar estas em outra pasta
i = 0
path = 'C:/Users/pietr/OneDrive/Área de Trabalho/Facul/2022-1/Processamento digital de sinais/TGB/treinamento_retinas/DATABASE_NORMAIS'
for filename in os.listdir(path):
    ims = cv2.imread('DATABASE_NORMAIS/normal'+str(i)+'.jpg')   # ler a imagem
    gray = cv2.cvtColor(ims, cv2.COLOR_BGR2GRAY)  #converter para GRAY
    
    #Gaussiano
    blur = cv2.GaussianBlur(gray,(3,3),0)
    imgray = blur
    
    # Thresh OTSU para definir a área desejada (círculo)
    thresh = cv2.threshold(imgray, 30, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    
    # Achando contorno da área
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Caixa delimitadora e Region Of Interest (cortando nos limites do círculo)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = ims[y:y+h, x:x+w]
        break
    ROIgray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    img = ROIgray
    
    # Uniformizar o tamanho e Inverter as imagens horizontalmente, verticalmente e em ambas direções
    imf = cv2.resize(img, (1000, 900))
    flipVertical = cv2.flip(imf, 0)
    flipHorizontal = cv2.flip(imf, 1)
    flipAmbos = cv2.flip(imf, -1)
    
    # Aplicando CLAHE na original e nas invertidas
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl1 = clahe.apply(imf)
    cl2 = clahe.apply(flipVertical)
    cl3 = clahe.apply(flipHorizontal)
    cl4 = clahe.apply(flipAmbos)
    
    # Salvando a imagem final e as suas versões invertidas na pasta para treinar a rede neural
    cv2.imwrite('C:/Users/pietr/OneDrive/Área de Trabalho/Facul/2022-1/Processamento digital de sinais/TGB/treinamento_retinas/RESULTADO_NORMAIS/norFinal'+str(i)+'.jpg', cl1)
   # cv2.imwrite('C:\Users\pietr\OneDrive\Área de Trabalho\Facul\2022-1\Processamento digital de sinais\TGB\treinamento_retinas\RESULTADO_NORMAIS/norFinal_fv'+str(i)+'.jpg', cl2)
    #cv2.imwrite('C:\Users\pietr\OneDrive\Área de Trabalho\Facul\2022-1\Processamento digital de sinais\TGB\treinamento_retinas\RESULTADO_NORMAIS/norFinal_fh'+str(i)+'.jpg', cl3)
    #cv2.imwrite('C:\Users\pietr\OneDrive\Área de Trabalho\Facul\2022-1\Processamento digital de sinais\TGB\treinamento_retinas\RESULTADO_NORMAIS/norFinal_fa'+str(i)+'.jpg', cl4)
    
    i = i + 1


#cv2.imshow('Original Gray', img)
#cv2.imshow('CLAHE', cl1)