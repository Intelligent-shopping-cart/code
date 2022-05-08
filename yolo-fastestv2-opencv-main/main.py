import cv2
import numpy as np
import argparse
import onnxruntime
from kcf import Tracker
class yolo_fast_v2():
    def __init__(self, objThreshold=0.3, confThreshold=0.3, nmsThreshold=0.4):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')   ###这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.stride = [16, 32]
        self.anchor_num = 3
        self.anchors = np.array([12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
                           dtype=np.float32).reshape(len(self.stride), self.anchor_num, 2)
        self.inpWidth = 352
        self.inpHeight = 352
        self.net =onnxruntime.InferenceSession('model.onnx',providers=['CUDAExecutionProvider'])
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            if classId!=0:
                continue
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence*detection[4]))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        outbox=[]
        for i in indices:
            box = boxes[i]
            outbox.append(box)
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame,outbox

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        #only person
        if classId!=0:
            return frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame
    def detect(self, srcimg):
        output_tensor=[node.name for node in self.net.get_outputs()]
        input_tensor=self.net.get_inputs()
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        outs=self.net.run(output_tensor,input_feed={input_tensor[0].name:blob})[0]
        #outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        outputs = np.zeros((outs.shape[0]*self.anchor_num, 5+len(self.classes)))
        row_ind = 0
        for i in range(len(self.stride)):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(h * w)
            grid = self._make_grid(w, h)
            for j in range(self.anchor_num):
                top = row_ind+j*length
                left = 4*j
                outputs[top:top + length, 0:2] = (outs[row_ind:row_ind + length, left:left+2] * 2. - 0.5 + grid) * int(self.stride[i])
                outputs[top:top + length, 2:4] = (outs[row_ind:row_ind + length, left+2:left+4] * 2) ** 2 * np.repeat(self.anchors[i, j, :].reshape(1,-1), h * w, axis=0)
                outputs[top:top + length, 4] = outs[row_ind:row_ind + length, 4*self.anchor_num+j]
                outputs[top:top + length, 5:] = outs[row_ind:row_ind + length, 5*self.anchor_num:]
            row_ind += length
        return outputs
import numpy


def yolodect(srcimg,roi):

    outputs=model.detect(srcimg)
    srcimg ,boxs= model.postprocess(srcimg, outputs)
    if len(boxs)==0:
        return srcimg,None
    matched=matchbox(boxs, roi)

    return srcimg,matched
def matchbox(boxs,mroi):
    #return matched box
    roict=[mroi[0]+(mroi[2])/2,mroi[1]+(mroi[3])/2]
    center=[]
    dis=[]
    for i in boxs:
        center.append([i[0]+(i[2]/2),i[1]+(i[3]/2)])
    for i in center:
        dit=(i[0]-roict[0])*(i[0]-roict[0])+(i[1]-roict[1])*(i[1]-roict[1])
        dis.append(dit)
    matchindex=np.argmin(dis)
    return boxs[matchindex]


if __name__ == '__main__':
    import torch
    import time
    from kalmanfilter import KalmanFilter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 保存视频的编码
    out = cv2.VideoWriter('turn left.MP4', fourcc, 30.0, (480, 640))  # 保存的格式
    cap=cv2.VideoCapture('turnleft.mp4')
    ret,srcimg=cap.read()
    model = yolo_fast_v2(objThreshold=0.3, confThreshold=0.3,
                         nmsThreshold=0.4)
    tracker = Tracker()
    miss=False
    while ret:

        ret, srcimg = cap.read()
        srcimg=cv2.resize(srcimg,(640,480))
        s=time.time()
        inum=cap.get(1)
        if inum==2:
            #初始化kcf tracker
            roi = cv2.selectROI("1", srcimg, False, False)
            srcimg, match = yolodect(srcimg, roi)
            tracker.init(srcimg, roi)
        if miss==True:
            srcimg, match = yolodect(srcimg, roi)
            if match ==None:
                miss = True
                continue
            miss=False
            tracker.init(srcimg, match)
        if (inum+1)%1==0:
            srcimg,match=yolodect(srcimg,roi)
            if match==None:
                miss=True
            if miss==True:
                x, y, w, h = tracker.update(srcimg)
                cv2.rectangle(srcimg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                tracker.init(srcimg,match)
        else:
            x, y, w, h = tracker.update(srcimg)
            cv2.rectangle(srcimg, (x, y), (x + w, y + h), (0,255, 0), 2)
        e=time.time()
        cv2.putText(srcimg, 'fps:{}'.format(int(1/(e-s))), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('1', srcimg)
        out.write(srcimg)
        cv2.waitKey(1)
