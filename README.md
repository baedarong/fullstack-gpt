# FullStack GPT

## Need Skills

1. Langchain / Langsmith LLM framework
2. Streamlit for python
3. Pinecone for vector
4. FastAPI
5. Hugging Face for other models
6. Virtual Environment setting

```
#python3 install
brew install pyenv
sudo apt-get update && sudo apt-get upgrade
pyenv install 3.11.6
pyenv global 3.11.6

#making virtual environment
python3 -m venv ./env

#get into virtual environment
source env/bin/activate

#install package
pip install -r requirements.txt

#get out from virtual environment
deactivate
```

## LANGCHAIN

### LLMs and Chat Models

랭체인의 장점 - 각 모델 API를 알아야 할 필요 없고, 모델을 제공하는 기업을 알 필요도 없다. 그냥 import 해서 사용하면 된다. 다른 모든 것과 호환할 수 있다.

```
# 아래와 같이 패키지 형식으로 import 하여 사용하면 각 모델의 API를 알 필요도 없다!
from  langchain.chat_models  import  ChatOpenAI
chat  =  ChatOpenAI()
chat.predict("How many planets are there?")
```

### Predict Messages

Pass a message sequence to the model and return a message prediction.

```
# SystemMessage: A Message for priming AI behavior
from  langchain.schema  import  HumanMessage, AIMessage, SystemMessage

messages  = [
SystemMessage(content='you are a geography expert. and you only reply in korean.'),
AIMessage(content='무엇이 궁금하신가요? 저는 지리학 박사입니다!'),
HumanMessage(content='한국과 일본의 지리적 차이에 대해 알려줘.')
]

# returns model prediction as a message.
chat.predict_messages(messages)
```
