from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from summarizer_critic_agent import load_agent

agent = load_agent()
app = FastAPI()

@app.get("/")
def run():
    return{
        'message': 'Hi'
    }

@app.get('/status')
def status():
    return{
        'status': 'Running'
    }

@app.post('/summarizer/chat')
def get_agent(input: str):

    res = agent.invoke(
        {
            'messages': HumanMessage(content=input),
            'input': input,
            'counter': 0,
            'threshold': 9
        }
    )

    return {
        'message': res['summary']
    }

