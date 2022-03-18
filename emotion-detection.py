# emotion_detection.py

# OpenCv is a python library designed to solve computer vision problems
import cv2
# DeepFace is a powerful computer vision library that’s helpful in identifying things in images
# (detect and analyse them)
from deepface import DeepFace

# image path
imgPath = 'image.png'

# load the image
image = cv2.imread(imgPath)

# the analyse attribute can tell us about the age, gender, race and  facial expression from the provided image
obj = DeepFace.analyze(img_path=imgPath, actions=['age', 'gender', 'race', 'emotion'])


analyze = DeepFace.analyze(image, actions=['emotion'])

print(analyze['dominant_emotion'])
