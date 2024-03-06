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

LangChain은 자연어 처리(NLP)를 위한 프레임워크로, 특히 대화형 AI 시스템을 구축할 때 유용합니다. LangChain의 장점은 다음과 같습니다:

1.  모듈화 및 확장성: LangChain은 여러 NLP 작업을 위한 모듈을 제공하며, 이를 조합하여 복잡한 시스템을 구축할 수 있습니다. 사용자는 필요에 따라 새로운 모듈을 쉽게 추가하거나 기존 모듈을 수정할 수 있습니다.
2.  대화 관리: LangChain은 대화 상태 관리, 대화 흐름 제어, 멀티턴 대화를 지원하는 기능을 제공하여 사용자가 자연스러운 대화형 AI를 구축할 수 있도록 돕습니다.
3.  통합된 API: 다양한 NLP 서비스와 툴을 하나의 프레임워크 내에서 쉽게 통합할 수 있어, 개발자가 여러 서비스를 동시에 관리하고 사용하는 데 드는 노력을 줄일 수 있습니다.
4.  커뮤니티 및 지원: LangChain은 오픈소스 프로젝트로, 개발자 커뮤니티의 지원을 받으며 지속적으로 발전하고 있습니다. 이는 문제 해결, 새로운 기능 추가, 최신 NLP 기술의 통합 등에 도움이 됩니다.
5.  유연성: LangChain은 다양한 언어와 NLP 엔진을 지원하므로, 개발자는 특정 언어나 기술에 제한되지 않고 자신의 요구에 맞는 시스템을 구축할 수 있습니다.
6.  빠른 프로토타이핑: LangChain을 사용하면 복잡한 NLP 기능을 빠르게 프로토타입으로 만들 수 있어, 아이디어를 신속하게 테스트하고 반복 개발할 수 있습니다.
7.  최신 AI 기술 활용: LangChain은 최신 AI 및 NLP 기술을 쉽게 통합할 수 있도록 설계되어 있어, 개발자가 최신 연구 결과와 도구를 활용하여 경쟁력 있는 솔루션을 만들 수 있습니다.
8.  비용 효율성: 오픈소스 프레임워크인 LangChain을 사용함으로써, 개발 비용을 절감하고, 라이선스 비용 없이 소프트웨어를 사용할 수 있습니다.

이러한 장점들은 LangChain을 사용하여 대화형 AI, 챗봇, 가상 비서 등을 개발하는 데 매우 유용하게 만듭니다.

### LLMs and Chat Models

```
# 아래와 같이 패키지 형식으로 import 하여 사용하면 각 모델의 API를 자세히 알 필요 없다!
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI

chat = ChatOpenAI()
chat.predict("How many planets are there?")

llm = OpenAI(model_name="gpt-3.5-turbo-1106")
question = llm.predict("How many planets are there?")
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

### Prompt Templates

Create a chat prompt template from a variety of message formats.

```
### Prompt Templates

from  langchain.chat_models  import  ChatOpenAI
from  langchain.callbacks  import  StreamingStdOutCallbackHandler

chat = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()]) ## call Chat large language models API.

# Load a prompt template from a template.
template = PromptTemplate.from_template('{country_a}과 {country_b}의 지리적 차이에 대해 알려줘.')

prompt = template.format(
	country_a='korea',
	country_b='japan')

# Pass a single string input to the model and return a string prediction.
chat.predict(prompt)

# Create a chat prompt template from a variety of message formats.
template = ChatPromptTemplate.from_messages ([
	('system', 'you are a geography expert. and you only reply in {language}.'),
	('ai', '무엇이 궁금하신가요? 저는 {name}입니다!'),
	('human', '{country_a}과 {country_b}의 지리적 차이에 대해 알려줘.')
])

prompt = template.format_messages(language="korean", name='배다롱', country_a='한국', country_b='일본')

#Pass a message sequence to the model and return a message prediction.
chat.predict_messages(prompt)
```

### OutputParser

Output parsers help structure language model responses.

```
from  langchain.schema  import  BaseOutputParser

# Output parsers help structure language model responses.
class  CommaOutputParser(BaseOutputParser):
# Parse a single string model output into some structure.
	def  parse(self,text):
		items= text.strip().split(',')
		return  list(map(str.strip, items))

p = CommaOutputParser()
p.parse("hello, how, are, you?")
# returns ['hello', 'how', 'are', 'you?']
```

### LCEL

LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.
https://js.langchain.com/docs/expression_language/interface

```
template = ChatPromptTemplate.from_messages([
	("system", "you are a list generating machine. everything you are asked will be answered with a comma separated list of max {max_items} in lowercase. do not reply with anything else."),
	("human", "{question}"),
])

# Basic example: prompt + model + output parser
# The `|` symbol chains together the different components feeds the output from one component as input into the next component.
chain = template  |  chat  |  CommaOutputParser()
chain.invoke({"max_items":5, "question": "flowers"})
```

### Chaining Chains

Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. The primary supported way to do this is with [LCEL](https://python.langchain.com/docs/expression_language).

```
chef_template = ChatPromptTemplate.from_messages([
	("system", "you are a world-class international chef. you create easy to follow recipies for any typeof cuisins with easy to find ingredients."),
	("human", "I want to cook {cuision} food."),
])

chef_chain = chef_template  |  chat

veg_chef_template = ChatPromptTemplate.from_messages([
	("system", "You are a vegetarian chef specialized on making traditional recipies vegetarian. You find alternative ingredients and explain their preparation. You don't radically modify the recipe. If there is no alternative for a food just say you don't know how to replace it."),
	("human", "{recipe}"),
])

veg_chain = veg_chef_template  |  chat

final_chain = {"recipe": chef_chain} | veg_chain
final_chain.invoke({'cuision':'indian'})
```
