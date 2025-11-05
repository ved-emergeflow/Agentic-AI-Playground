from fastapi import FastAPI
from summarizer_critic_agent import load_agent

app = FastAPI()

@app.get('/status')
def status():
    return{
        'message': 'Status: Running'
    }

@app.get('/summarizer')
def get_agent():

    agent = load_agent()
