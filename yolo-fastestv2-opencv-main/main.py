import math
import cv2
import numpy as np
import onnxruntime
from kcf import Tracker
from phash import pHash,cmpHash

class yolo_fast_v2():
    def __init__(self, objThreshold=0.3, confThreshold=0.3, nmsThreshold=0.4):
        with open('coco.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
            # 这个是在coco数据集上训练的模型做opencv部署的，如果你在自己的数据集上训练出的模型做opencv部署，那么需要修改self.classes
        self.stride = [16, 32]
        self.anchor_num = 3
        self.anchors = np.array(
            [12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
            dtype=np.float32).reshape(len(self.stride), self.anchor_num, 2)
        self.inpWidth = 352
        self.inpHeight = 352
        self.net = onnxruntime.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])
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
            if classId != 0:
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
                confidences.append(float(confidence * detection[4]))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        outbox = []
        for i in indices:
            box = boxes[i]
            outbox.append(box)
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame, outbox

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        # only person
        if classId != 0:
            return frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        #cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)
        return frame

    def detect(self, srcimg):
        output_tensor = [node.name for node in self.net.get_outputs()]
        input_tensor = self.net.get_inputs()
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight))
        outs = self.net.run(output_tensor, input_feed={input_tensor[0].name: blob})[0]
        # outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        outputs = np.zeros((outs.shape[0] * self.anchor_num, 5 + len(self.classes)))
        row_ind = 0
        for i in range(len(self.stride)):
            h, w = int(self.inpHeight / self.stride[i]), int(self.inpWidth / self.stride[i])
            length = int(h * w)
            grid = self._make_grid(w, h)
            for j in range(self.anchor_num):
                top = row_ind + j * length
                left = 4 * j
                outputs[top:top + length, 0:2] = (outs[row_ind:row_ind + length,
                                                  left:left + 2] * 2. - 0.5 + grid) * int(self.stride[i])
                outputs[top:top + length, 2:4] = (outs[row_ind:row_ind + length,
                                                  left + 2:left + 4] * 2) ** 2 * np.repeat(
                    self.anchors[i, j, :].reshape(1, -1), h * w, axis=0)
                outputs[top:top + length, 4] = outs[row_ind:row_ind + length, 4 * self.anchor_num + j]
                outputs[top:top + length, 5:] = outs[row_ind:row_ind + length, 5 * self.anchor_num:]
            row_ind += length
        return outputs


def boxdistance(box):
    box1 = box[0]
    box2 = box[1]
    mid_point1_x = (box1[0] + box1[2]) / 2
    mid_point1_y = (box1[1] + box1[3]) / 2
    mid_point2_x = (box2[0] + box2[2]) / 2
    mid_point2_y = (box2[1] + box2[3]) / 2
    distance = math.sqrt(math.pow((mid_point2_x-mid_point1_x), 2)+math.pow((mid_point2_y-mid_point1_y), 2))
    return distance


def yolodect(srcimg, roi):
    outputs = model.detect(srcimg)
    srcimg, boxs = model.postprocess(srcimg, outputs)

    if len(boxs) == 0:
        return srcimg, None, 1, 0
    matched = matchbox(boxs, roi)
    if len(boxs) == 2:
        distance = boxdistance(boxs)
        return srcimg, matched, distance, len(boxs),boxs
    return srcimg, matched, 1, len(boxs),boxs   #boxs is added by JLH in 5.10 use for match person


def matchbox(boxs, mroi):
    # return matched box
    roict = [mroi[0] + (mroi[2]) / 2, mroi[1] + (mroi[3]) / 2]
    center = []
    dis = []
    for i in boxs:
        center.append([i[0] + (i[2] / 2), i[1] + (i[3] / 2)])
    for i in center:
        dit = (i[0] - roict[0]) * (i[0] - roict[0]) + (i[1] - roict[1]) * (i[1] - roict[1])
        dis.append(dit)
    matchindex = np.argmin(dis)
    return boxs[matchindex]


def PCB(img):
    h,w=img.shape[:2]
    header = h / 7
    part1 = h * 4 / 7
    p1=img[int(header):int(part1),:]
    p2=img[int(part1):h, :]
    return p1,p2
def sim(p1,p2,ip1,ip2):
    p1=pHash(p1)
    p2=pHash(p2)
    sim=cmpHash(p1, ip1)*0.6+0.4*cmpHash(p2,ip2)
    return  sim

def matchperson(roi,img1,imgmatching):
    simlist=[]
    imgmatching=cv2.GaussianBlur(imgmatching, (5, 5), 3)
    ip1,ip2=PCB(imgmatching)
    ip1=pHash(ip1)
    ip2=pHash(ip2)
    for i in roi:
        if i[2]*i[3]<2048: #过滤面积较小的roi
            simlist.append(100)
            continue
        p=img1[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
        p = cv2.GaussianBlur(p, (5, 5), 3)
        p1,p2=PCB(p)
        s=sim(p1,p2,ip1,ip2)
        #cv2.imshow('2',p)
        #cv2.waitKey(0)
        simlist.append(s)
    return np.argmin(simlist)
def overlap(dis_list, nbox_list, high):
    '''
    判断目标是否重叠
    :param dis_list: 存储bbox距离的list
    :param nbox_list: 存储bbox数量的list
    :param high:距离上限阈值
    :return: dis_list,nbox_list
    '''
    if len(dis_list) == 6:
        for i in range(0, 4):
            if nbox_list[i + 1] == nbox_list[i] - 1 and 1 < dis_list[i] < high:
                roi = cv2.selectROI("1", srcimg, False, False)
        if nbox_list[-1] >= 2:
            nbox_list = []
            dis_list = []
            nbox_list.append(2)
            dis_list.append(high)
        else:
            nbox_list = []
            dis_list = []
        return dis_list, nbox_list
    else:
        return dis_list, nbox_list
if __name__ == '__main__':
    import time

    cap = cv2.VideoCapture(r'./data/outside1.mp4')
    ret, srcimg = cap.read()
    model = yolo_fast_v2(objThreshold=0.3, confThreshold=0.3, nmsThreshold=0.4)
    tracker = Tracker()
    miss = False
    dis_list = []
    num_box_list = []
    tmpimg=np.zeros_like(srcimg)
    while ret:
        ret, srcimg = cap.read()
        srcimg = cv2.resize(srcimg, (640, 480))
        s = time.time()
        inum = cap.get(1)
        if inum == 2:
            # 初始化kcf tracker 和匹配对象
            roi = cv2.selectROI("1", srcimg, False, False)
            tracker.init(srcimg, roi)
            tmpimg=srcimg[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
        if (inum + 1) % 1 == 0:
            srcimg, match, distance, num_box,boxs = yolodect(srcimg, roi)
            boxindex=matchperson(boxs,srcimg,tmpimg)
            tmpimg=srcimg[boxs[boxindex][1]:boxs[boxindex][1]+boxs[boxindex][3],boxs[boxindex][0]:boxs[boxindex][0]+boxs[boxindex][2]]
            cv2.rectangle(srcimg,(boxs[boxindex][0],boxs[boxindex][1]),(boxs[boxindex][0]+boxs[boxindex][2],boxs[boxindex][1]+boxs[boxindex][3]),(255,255,0),3)
            if inum % 1== 0:
                dis_list.append(distance)
                num_box_list.append(num_box)
                dis_list,num_box_list=overlap(dis_list,num_box_list,35)

        e = time.time()
        cv2.putText(srcimg, 'fps:{}'.format(int(1 / (e - s))), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('1', srcimg)
        c = cv2.waitKey(1)
