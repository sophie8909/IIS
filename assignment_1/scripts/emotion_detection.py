import cv2
from feat import Detector
from IPython.display import Image
from feat.utils import FEAT_EMOTION_COLUMNS


detector = Detector(device="cuda")

import os

# Construct the path to the image folder
ori_folder_path = os.path.join( '..', 'dataset', 'images')


# load all the images in the folder
images_filename = os.listdir(ori_folder_path)




# loop through all the images
for image_filename in images_filename:
    frame = cv2.imread(ori_folder_path + "/" + image_filename)
    print(ori_folder_path + "/" + image_filename)
    faces = detector.detect_faces(frame)
    landmarks = detector.detect_landmarks(frame, faces)
    emotions = detector.detect_emotions(frame, faces, landmarks)
    AUs = detector.detect_aus(frame, landmarks)
    faces = faces[0]
    landmarks = landmarks[0]
    emotions = emotions[0]

    strongest_emotion = emotions.argmax(axis=1)
    print(emotions)

    # draw rectangles around the faces
    for (face, top_emo) in zip(faces, strongest_emotion):
        (x0, y0, x1, y1, p) = face
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
        cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    
    
    
    
    cv2.imshow("frame", frame)
    results_file_path = os.path.join('..', 'processed', 'images', image_filename)
    cv2.imwrite(results_file_path, frame)
# save the results to a csv file
results_file_path = os.path.join('..', 'processed', 'aus.csv')