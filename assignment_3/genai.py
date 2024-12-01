from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
from genai_api_key import GENAI_API_KEY 

furhat = FurhatRemoteAPI("localhost")
voices = furhat.get_voices()


genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
