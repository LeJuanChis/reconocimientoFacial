import os
import cv2
import imutils
import numpy as np

def emotionImage(emotion):
    image = ''
    if emotion == 'Felicidad': image = cv2.imread("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/emotions/Felicidad.jpeg")
    if emotion == 'Enojo': image = cv2.imread("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/emotions/Enojo.jpeg")
    if emotion == 'Tristeza': image = cv2.imread("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/emotions/Tristeza.jpeg")
    if emotion == 'Sorpresa': image = cv2.imread("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/emotions/Sorpresa.jpeg")
    return image

dataPath = "C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/emotions"
imagePath = os.listdir(dataPath)

print("paths = {}".format(imagePath))

cap = cv2.VideoCapture(0)
cap.open(0)

#Metodos para reconocimiento y entenamiento
# method = "EigenFaces"
# method = "FisherFaces"
method = "EigenFaces"

if method == 'EigenFaces': face_recognize = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': face_recognize = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': face_recognize = cv2.face.LBPHFaceRecognizer_create()

#leer el modelo
face_recognize.read("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/trainEmotions/modeloReconocimiento"+method+".xml")

facesClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

while True:
    res, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copyFrame = gray.copy()
    newFrame = cv2.hconcat([frame, np.zeros((480,300,3), dtype=np.uint8)])


    faces = facesClassif.detectMultiScale(gray,
    scaleFactor= 1.1,
    minNeighbors= 3
    )

    for (x,y,w,h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = copyFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognize.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        if method == 'EigenFaces':
            #EigenFaces
            if result[1] <= 5700:
                cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
                image = emotionImage(imagePath[result[0]].split('.')[0])
                newFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, '{}'.format('Desconocido'), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif method == 'FisherFaces':
            #Fisher Faces
            if result[1] <= 500:
                cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
                image = emotionImage(imagePath[result[0]].split('.')[0])
                newFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame, '{}'.format('Desconocido'), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        elif method == 'LBPH':
            #LBPH
            if result[1] <= 70:
                cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
                image = emotionImage(imagePath[result[0]].split('.')[0])
                newFrame = cv2.hconcat([frame, image])
            else:
                cv2.putText(frame, '{}'.format('Desconocido'), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('nframe', newFrame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



