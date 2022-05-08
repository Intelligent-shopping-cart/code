import argparse
import cv2
from kcf import Tracker

if __name__ == '__main__':
    import time
    cap = cv2.VideoCapture(r'cross.mp4')
    tracker = Tracker()
    ok, frame = cap.read()
    frame=cv2.resize(frame,(640,480))
    if not ok:
        print("error reading video")
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)
    #roi = (218, 302, 148, 108)
    tracker.init(frame, roi)
    while cap.isOpened():
        ok, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        s=time.time()
        if not ok:
            break
        x, y, w, h = tracker.update(frame)
        e=time.time()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c==27 or c==ord('q'):
                roi = cv2.selectROI("tracking", frame, False, False)
    cap.release()
    cv2.destroyAllWindows()
