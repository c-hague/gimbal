import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import numpy as np
from SunFounder_PCA9685 import Servo
import RPi.GPIO as GPIO

TEMP_FILE = 'temp.jpg'
ALL_OFF_PIN = 11

INTEGRAL_HISTORY = 10
KP = 1
KI = 0
KD = 0

def main():
    try:
        print('starting ctrl+C to exit...')
        # emergency shuttoff
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(ALL_OFF_PIN, GPIO.OUT)
        GPIO.output(ALL_OFF_PIN, GPIO.HIGH)
        # enable the PC9685 and enable autoincrement
        pan = Servo.Servo(1, bus_number=1)
        tilt = Servo.Servo(0, bus_number=1)
        tilt.write(90)
        pan.write(90)
        # camera and open cv
        picSpace = (640, 480)
        camera = PiCamera()
        camera.resolution = picSpace
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=picSpace)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # pid variables
        history = [np.array([[0], [0], [0]])] * INTEGRAL_HISTORY
        # allow the camera to warmup
        time.sleep(0.1)
        # capture frames from the camera
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            img = frame.array
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found = faces.detectMultiScale(gray, minSize =(20, 20))
            if len(found) != 0:
                x, y, width, height = found[0]
                # cv2.rectangle(img, (x, y), 
                #             (x + height, y + width), 
                #             (0, 255, 0), 5)
                # center of face
                cX = x + width / 2
                cY = y + height / 2
                # transform coordinate system
                cX = cX - picSpace[0]
                cY = (cY - picSpace[1]) * -1

                error, integral, derivative, dt = calcErrorTerms(cX, cY, time.time(), history)
                #  print('error terms P ({0},{1}) I ({2},{3}) D ({4},{5})'.format(error[0], error[1], integral[0], integral[1], derivative[0], derivative[1]))
                pid = KP * error + KI * integral + KD * derivative
                pan.write(pid[0])
                tilt.write(pid[1])
                print(pid)
            # cv2.imwrite(TEMP_FILE, img)
            else:
                print('no face found!')
                pan.write(90)
                tilt.write(90)
            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
    finally:
        pan.write(0)
        tilt.write(0)
        GPIO.cleanup()

def calcErrorTerms(x, y, t, history):
    history.append(np.array([[x], [y], [t]]))
    history.pop(0)
    intg = np.zeros([2,1])
    for i in range(len(history) - 1):
        dt = history[i + 1][2] - history[i][2]
        # left sum
        intg = history[i][:2] * dt + intg
    der = (history[-1][:2] - history[-2][:2]) / (history[-1][2] - history[-2][2])
    return np.array([[x], [y]]), intg, der, dt

if __name__ == '__main__':
        main()
