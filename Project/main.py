from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from genai_api_key import GENAI_API_KEY 
from feat import Detector
from feat.pretrained import AU_LANDMARK_MAP
import random



if __name__ == "__main__":
    detector = Detector(device="cuda")
    furhat = FurhatRemoteAPI("localhost")
    voices = furhat.get_voices()


    genai.configure(api_key=GENAI_API_KEY)
    # Prompt to GEMINI model presonality
    # read the text from prompt/person1.txt
    with open("prompt/person1.txt", "r") as file:
        instruction = file.read()
    model = genai.GenerativeModel('gemini-1.5-flash-latest', system_instruction=instruction)
    response = model.generate_content("hi")


    # Select a character for the virtual Furhat
    furhat.set_face(character="AnimePink", mask="Anime")

    # Set the voice of the robot
    furhat.set_voice(name='Joanna')

    # Have Furhat greet the user
    furhat.say(text=response.text, blocking=True)

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



    # Listen to the user's response
    user_response = furhat.listen()  
    while user_response.success:
        
        # Check if listening was successful
        if user_response.success and user_response.message:

            print("User said:", user_response.message)
            response = model.generate_content(user_response.message)
            print("Furhat said:", response.text)
            furhat.say(text=response.text, blocking=True)
            
        else:
            if user_response.message == "":
                pass
            furhat.say(text="Sorry, I cannot understand", blocking=True)
            print("Listening failed or no speech detected.")
        user_response = furhat.listen()  
