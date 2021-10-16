import face_recognition
import numpy as np
import os
import cv2

imgElon = face_recognition.load_image_file('Basic_Images/elon_musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Basic_Images/elon_musk_2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
faceEncode = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
faceEncodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([faceEncode],faceEncodeTest)
faceDist = face_recognition.face_distance([faceEncode],faceEncodeTest)
print(results, faceDist)

cv2.imshow('Image', imgElon)
cv2.imshow('Test image', imgTest)
cv2.waitKey(0)


