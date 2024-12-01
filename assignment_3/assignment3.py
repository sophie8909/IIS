from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from genai_api_key import GENAI_API_KEY 
import random

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


furhat_gesture_list = ["Nod", "Smile", "Blink", "Shake",  "Surprise", "Wink"]


# Listen to the user's response
user_response = furhat.listen()  
while user_response.success:
    # Check if listening was successful
    if user_response.success and user_response.message:
        gesture = random.choice(furhat_gesture_list)
        furhat.gesture(name=gesture)    
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
