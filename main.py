import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time


TEMP_FILE = 'temp.jgp'


def main():
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # allow the camera to warmup
    time.sleep(0.1)
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BRG2GRAY)
        found = faces.detectMultiScale(frame, minSize =(20, 20))        
        if len(found) != 0:
            x, y, width, height = found[0]
            cv2.rectangle(frame, (x, y), 
                        (x + height, y + width), 
                        (0, 255, 0), 5)
            cv2.imwrite(TEMP_FILE, frame)

        # wait 1ms for key to see if quitting is needed
        key = cv2.waitKey(1) & 0xFF
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break