import gradio as gr
import random
from graph import app, memory

sessions = {}
ids = set()

def same_auth(username, password):
    user = username

    if user in sessions.keys():
        print("User already exist !!")
    
    else:
        config_id = random.randint(1, 100)
        
        while(config_id in ids):
            config_id = random.randint(1, 100)

        sessions[user] = config_id
        ids.add(config_id)
        
        with open("user_ids", "a") as file:
            file.write(f"{user} ----> {config_id}\n")

        print("User created !!")

    return username == password

def chatbot_response(user_input, history, request: gr.Request):

    if request:
        user = request.username

    config_id = sessions[user]
    
    response = ""
   
    config = {"configurable": {"thread_id": config_id}}

    event = app.invoke({"input_msg": user_input}, config)

    response = event['translated_generation']

    return response

iface = gr.ChatInterface(chatbot_response)
# Launch the Gradio app
iface.launch(auth = same_auth)