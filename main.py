from flask import Flask, render_template, request, redirect
import cv2
import imutils
import time
import threading
import pyttsx3
import os




#function to get the traning to the frontal face recognizer
def camera(personName = '', emotionName = ''):

    dataPath = "C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/emotions"
    personPath = dataPath + '/' + emotionName
    vid = cv2.VideoCapture(0)
    vid.open(0)
    if not os.path.exists(personPath):
        os.makedirs(personPath)

    #vid = cv2.VideoCapture("C:/Users/Janus/Desktop/Aprendiendo python/aprendiendo-python/webPractice/data/train/mery.mp4")
    facesClassif = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    count = 0

    while True:
        res, frame = vid.read()
        # if res== False : break
        # frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        copyFrame = frame.copy()

        faces = facesClassif.detectMultiScale(frame,
        scaleFactor= 1.2,
        minNeighbors= 3
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rostro = copyFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_ {}.jpg'.format(count),rostro)
            count= count + 1

        cv2.imshow('frame', frame)
        key= cv2.waitKey(1)
        if key == 27 or 0xff == ord('q') or count >= 800:
            break
    vid.release()
    cv2.destroyAllWindows()

#voice function
def speak(text):
    engine = pyttsx3.init()
    voices=engine.getProperty('voices')
    #0= gringo  1=ingles 2= español de españa
    engine.setProperty('voice', voices[2].id)
    engine.say(text)
    engine.runAndWait()

# create the flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():

    return render_template("index.html")

@app.route('/faceDetection', methods=['POST'])
def faceDetection():
    _name = request.form['name']
    _emotion = request.form['emotion']
    speak(f"Hola {_name} como estas? La camara se va a prender, necesitamos que mantengas tu mirada fija en ella haciendo expresiones faciales")
    # ejecute with subprocess
    cam_thread = threading.Thread(target=camera(_name, _emotion))
    cam_thread.start()
    return redirect('/')

if __name__ == '__main__':
    app.run(port=5990, debug=True)
