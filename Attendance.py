import face_recognition
import numpy as np
import os
import cv2
from datetime import datetime

path = 'Attendance_Images'
Images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    Images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)


# To find Encodings
def FindEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def mark_Attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        #print(myDataList)
        namelist= []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtStr}')


encodeKnownImages = FindEncodings(Images)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faceCurImg = face_recognition.face_locations(img_small)
    encodeCurImg = face_recognition.face_encodings(img_small)

    for encodeFace, faceloc in zip(encodeCurImg, faceCurImg):
        matches = face_recognition.compare_faces(encodeKnownImages, encodeFace)
        faceDis = face_recognition.face_distance(encodeKnownImages, encodeFace)
        # print(faceDis)
        # The one with a match will have the least distance between faces
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex]<0.5:
            name = classNames[matchIndex].upper()
            mark_Attendance(name)
        else:
            name = 'Unknown'
            # print(name)
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


        cv2.imshow('Webcam', img)
        cv2.waitKey(1)
