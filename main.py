import model_object.object_detector as od
import cv2

detector = od.ObjectDetector()
while True:
    detector.detectObjects()
    detector.check()
    detector.draw()  
    cv2.waitKey(1)
