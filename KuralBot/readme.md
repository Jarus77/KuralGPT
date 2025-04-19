# ThirukuralBot
ThirukuralBot is an intelligent chatbot designed to assist with inquiries about Thirukural, the Tamil classic by Thiruvalluvar. It provides users with instant insights on ethical teachings, practical wisdom, and guidance on virtue, wealth, and love.

## Instructions to run the app

1. Clone this repository using `git clone` command.

2. Make a virtual enviornment by executing following command in the working directory

    ```
    python -m venv <name_of_env>
    ```
    You can also use conda to make virtual env.

3. Activate the virual env using this command (not for conda env)- 
    ```
    source <name_of_env>/bin/activate
    ```

3. Install the libraries and packages using below command
   ```
   pip install -r requirements.txt
   ```

4. Make a `config.env` file and fill the following details in the file
    ```
    LANGCHAIN_API_KEY = <your_langchain_api_key>
    TAVILY_API_KEY = <your_tavily_web_search_api_key>
    LANGCHAIN_TRACING_V2 = true
    LANGCHAIN_PROJECT = <project_name>
    HF_API_KEY = <your_hugging_face_token>
    ```
5. Run the app.py using the command in terminal - `python app.py`. It will open a gradio app. Now you can chat with it.

## Disclaimer

This project is currently a work in progress. Features may change, and the application may not be fully stable. We are actively working on improvements, and your feedback is highly appreciated.

