import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import numpy as np
import smbus
import RPi.GPIO as GPIO

TEMP_FILE = 'temp.jpg'
ALL_OFF_PIN = 20
I2C_ADDR = 0x40
SERVO_PAN_LOW = 0x06
SERVO_PAN_HIGH = 0x08
SERVO_TILT_LOW = 0x0a
SERVO_TILT_HIGH = 0x0c

INTEGRAL_HISTORY = 10
KP = 1
KI = 0
KD = 0

def main():
    print('starting ctrl+C to exit...')
    # emergency shuttoff
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(ALL_OFF_PIN, GPIO.OUT)
    GPIO.output(ALL_OFF_PIN, GPIO.HIGH)
    # #I2C interface
    bus = smbus.SMBus(1)
    
    ## enable the PC9685 and enable autoincrement
    bus.write_byte_data(I2C_ADDR, 0, 0x20) # set mode 1
    bus.write_byte_data(I2C_ADDR, 0xfe, 0x1e) # set frequency

    bus.write_word_data(I2C_ADDR, 0x06, 0) # led0 output and brightness control byte 0
    bus.write_word_data(I2C_ADDR, 0x08, 1250) # led0 output and brightness control byte 2

    bus.write_word_data(I2C_ADDR, 0x0a, 0) #led1 output and brightness control byte 0
    bus.write_word_data(I2C_ADDR, 0x0c, 1250) #led1 output and brightness control byte 2

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
            print('error terms P ({0},{1}) I ({2},{3}) D ({4},{5})'.format(error[0], error[1], integral[0], integral[1], derivative[0], derivative[1]))
            pid = KP * error + KI * integral + KD * derivative
            pan(bus, pid[0])
            tilt(bus, pid[1])
        # cv2.imwrite(TEMP_FILE, img)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

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
    
def pan(bus, x):
    x = int(x)
    if x < -255:
        x = -255
    if x > 255:
        x = 255
    if x < 0:
        bus.write_byte_data(I2C_ADDR, SERVO_PAN_HIGH, 255 - x)
    else:
        bus.write_byte_data(I2C_ADDR, SERVO_PAN_LOW, x)

def tilt(bus, y):
    y = int(y)
    if y < -255:
        y = -255
    if y > 255:
        y = 255
    if y < 0:
        bus.write_byte_data(I2C_ADDR, SERVO_TILT_HIGH, 255 - y)
    else:
        bus.write_byte_data(I2C_ADDR, SERVO_TILT_LOW, y)
    
if __name__ == '__main__':
    try:
        main()
    finally:
        GPIO.output(ALL_OFF_PIN, GPIO.LOW)
        GPIO.cleanup()
