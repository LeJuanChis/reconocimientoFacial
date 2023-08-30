from time import time
import cv2
import os
import numpy as np
import time

def trainingAllMethods(method, facesData, labels):
    if method == 'EigenFaces': face_recognize = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': face_recognize = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': face_recognize = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando modelo ({})".format(method))
    inicio =time.time()
    #entrenar el modelo
    face_recognize.train(facesData, np.array(labels))
    #Almacenar el modelo
    face_recognize.write("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/trainEmotions/modeloReconocimiento" + method + ".xml")
    final = time.time() - inicio
    print("El modelo ha sido entrenado con {} segundos".format(final) )



dataPath = "C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/emotions"
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

trainingAllMethods('EigenFaces', facesData, labels)
# trainingAllMethods('FisherFaces', facesData, labels)
# trainingAllMethods('LBPH', facesData, labels)