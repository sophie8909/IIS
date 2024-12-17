from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from genai_api_key import GENAI_API_KEY 
import threading
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from feat import Detector
from feat.utils import FEAT_EMOTION_COLUMNS
import random
import json
import re
predefined_gestures = [
        "Smile",        # 微笑
        "BigSmile",     # 開懷大笑
        "Surprise",     # 表示驚訝
        "Nod",          # 點頭
        "Blink",        # 眨眼
        "Wink",         # 眨單眼
        "Thoughtful",   # 思考動作
        "BrowFrown",    # 皺眉
        "BrowRaise",    # 揚眉
        "CloseEyes",    # 閉眼
        "ExpressAnger", # 表示生氣
        "ExpressFear",  # 表示害怕
        "ExpressSad",   # 表示悲傷
        "ExpressDisgust", # 表示厭惡
        "GazeAway",     # 轉頭
        "Oh",           # 表示驚訝
        "OpenEyes",     # 睜眼
        "Roll",
        "Shake"
    ]

# Global variables
user_emotion = "neutral"


# Lock 
lock = threading.Lock()

# parse response as {'furhat_emotion': str, 'furhat_text': str, 'personality': str}
def parse_response(response):
    match = re.search(r'\{(.+?)\}', response.text, re.DOTALL)

    if match:
        extracted_content = match.group(0)  # 包含整個 {}，含括號
        print("json:", extracted_content)
    else:
        print("response format error")

    corrected_json = extracted_content.replace("'", "\\'")
    
    response = json.loads(extracted_content)
    return response

def temp_emotion_detect(frame):
    print("Starting emotion detection...")
    global user_emotion
    detector = Detector(device="cuda")
    cv2.namedWindow("Webcam Feed")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        landmarks = detector.detect_landmarks(frame, faces)
        emotions = detector.detect_emotions(frame, faces, landmarks)

        faces = faces[0]
        landmarks = landmarks[0]
        emotions = emotions[0]

        user_emotion = emotions.argmax(axis=1)


        # draw rectangles around the faces
        for (face, top_emo) in zip(faces, user_emotion):
            (x0, y0, x1, y1, p) = face
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
            cv2.putText(frame, FEAT_EMOTION_COLUMNS[top_emo], (int(x0), int(y0 - 10)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

            
        cv2.imshow("Webcam Feed", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

def emotion_detect():
    global user_emotion
    # 加載訓練好的模型
    emotion_detect_model = load_model('/Users/chenchenghan/Downloads/emotion_detection_fer2013.h5') 

    # 定義情緒標籤
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # 初始化 OpenCV 的人臉檢測分類器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 啟動攝像頭
    cap = cv2.VideoCapture(0)

    while True:
        # 從攝像頭捕捉影像
        ret, frame = cap.read()
        if not ret:
            break

        # 將影像轉換為灰階（因為情緒模型使用灰階影像）
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 偵測人臉
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # 繪製人臉邊框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # 裁剪人臉影像
            face = gray_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))  # 調整大小為模型需要的輸入
            face = face / 255.0  # 正規化
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)  # 添加通道維度 (48, 48, 1)

            # 使用模型進行預測
            emotion_predictions = emotion_detect_model.predict(face)
            max_index = np.argmax(emotion_predictions[0])  # 找到預測機率最高的類別
            with lock:
                user_emotion = emotion_labels[max_index]
            

                # 在人臉邊框上方輸出情緒文字
                cv2.putText(frame, user_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                print(f"Detected emotion: {user_emotion}")

        # 顯示影像
        cv2.imshow('Emotion Detection', frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()


def main():
    
    with open("./Prompt/INFP.txt", "r", encoding="utf-8") as file:
        INFP_instruction = file.read()
    # with open("./Prompt/ENTP.txt", "r", encoding="utf-8") as file:
    #     ENTP_instruction = file.read()
    # with open("./Prompt/ESFJ.txt", "r", encoding="utf-8") as file:
    #     ESFJ_instruction = file.read()
    genai.configure(api_key=GENAI_API_KEY)

    global furhat, INFP_model
    # , ENTP_model, ESFJ_model
    INFP_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=INFP_instruction)
    # ENTP_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=ENTP_instruction)
    # ESFJ_model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=ESFJ_instruction)

    furhat = FurhatRemoteAPI("localhost")

    print("Starting Furhat and emotion detection threads...")

    # 創建兩個 Thread
    thread_emotion = threading.Thread(target=temp_emotion_detect)

    # 啟動 Thread
    thread_emotion.start()

    print("Starting Furhat control...")
    
    
    voices = furhat.get_voices()

    # Select a character for the virtual Furhat
    furhat.set_face(character="AnimePink", mask="Anime")

    # Set the voice of the robot
    furhat.set_voice(name='Joanna')

    current_model = random.choice([INFP_model])
    #  ENTP_model, ESFJ_model
    change_model = False

    response = current_model.generate_content("the costumer is coming")
    response = parse_response(response)

    # Have Furhat greet the user
    furhat.say(text=response['furhat_text'], blocking=True)
    furhat.gesture(name=response['furhat_emotion'])


    user_response = furhat.listen()
    while user_response.success:
        # Check if listening was successful
        if user_response.success and user_response.message:
            # Get user current emotion string

            # user_current_emotion = 
            print("User said:", user_response.message)
            
            # Input: {'user_emotion': str, 'user_text': str}
            # Outpur: {'furhat_emotion': str, 'furhat_text': str, 'personality': str}
            # personality: 'ENTP', 'ESFJ', 'INFP'
            with lock:
                global user_emotion
                print("user_emotion:", user_emotion)
                print("{{'user_emotion': {}, 'user_text': {}}}".format(user_emotion, user_response.message))
                response = current_model.generate_content("{{'user_emotion': {}, 'user_text': {}}}".format(user_emotion, user_response.message))
            response = parse_response(response)
            
            print("response:", response)

            furhat_emotion = response['furhat_emotion']
            furhat_text = response['furhat_text']
            personality = response['personality']

            furhat.say(text=furhat_text, blocking=False)
            furhat.gesture(name=furhat_emotion)

            if personality == 'INFP' and current_model != INFP_model:
                change_model = True
                current_model = INFP_model
            # elif personality == 'ENTP' and current_model != ENTP_model:
            #     change_model = True
            #     current_model = ENTP_model
            # elif personality == 'ESFJ' and current_model != ESFJ_model:
            #     change_model = True
            #     current_model = ESFJ_model
            
        else:
            if user_response.message == "":
                pass
            print("Listening failed or no speech detected.")
        if change_model:
            current_model.generate_content("Your colleague is coming to talk to you. Since the costumer wants to talk to you, you should be prepared.")
            change_model = False
            furhat.say(text=response.text, blocking=True)
        user_response = furhat.listen()  
    


    

    
    
    

if __name__ == "__main__":
    main()