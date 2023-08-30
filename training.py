import cv2
import os
import numpy as np

dataPath = "C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/faces"
peopleList = os.listdir(dataPath)


labels = []
facesData = []
label = 0

for people in peopleList:
    facesPath = dataPath + '/' + people
    for nameImage in os.listdir(facesPath):
        labels.append(label)
        facesData.append(cv2.imread(facesPath + '/' + nameImage, 0)) #transformando a esala e grises
        image = cv2.imread(facesPath + '/' + nameImage, 0)
        # cv2.imshow('image', image)
        # cv2.waitKey(10)
    label = label + 1

#Entrenando el reconcoedor de rostros

# face_recognize = cv2.face.EigenFaceRecognizer_create() #crear un objeto de reconocimiento de engen face

face_recognize = cv2.face.FisherFaceRecognizer_create() #crear un objeto de reconcimiento de fisher face

#face_recognize = cv2.face.LBPHFaceRecognizer_create() #crear un objeto de reconocimeinto de LBPH

print("Entrenando...")
#entrenar el modelo
face_recognize.train(facesData, np.array(labels))

#Almacenar el modelo

face_recognize.write('C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/modeloReconocimientoFisher.xml')

print("El modelo ha sido entrenado")