import os
import cv2
import imutils

dataPath = "C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/faces"
imagePath = os.listdir(dataPath)

print("paths = {}".format(imagePath))

cap = cv2.VideoCapture(0)
cap.open(0)

# face_recognize = cv2.face.EigenFaceRecognizer_create()

face_recognize = cv2.face.FisherFaceRecognizer_create()

# face_recognize = cv2.face.LBPHFaceRecognizer_create()

#leer el modelo
face_recognize.read("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/modeloReconocimientoFisher.xml")

facesClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

while True:
    res, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    copyFrame = gray.copy()

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
        '''
        #EigenFaces
        if result[1] <= 5700:
            cv2.putText(frame, '{}'.format(imagePath[0]), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, '{}'.format('Desconocido'), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        '''
        #Fisher Faces
        if result[1] <= 500:
            cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, '{}'.format('Desconocido'), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        '''
        #LBPH
        if result[1] <= 70:
            cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, '{}'.format('Desconocido'), (x, y-25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        '''
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



