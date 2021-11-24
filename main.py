import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import logging

TEMP_FILE = 'temp.jpg'


def main():
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # allow the camera to warmup
    time.sleep(0.1)
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        img = frame.array
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = faces.detectMultiScale(gray, minSize =(20, 20))
        if len(found) != 0:
            x, y, width, height = found[0]
            cv2.rectangle(img, (x, y), 
                        (x + height, y + width), 
                        (0, 255, 0), 5)
        print('found {0} faces {1}'.format(len(found), time.time()))
        cv2.imwrite(TEMP_FILE, img)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)


if __name__ == '__main__':
    main()
