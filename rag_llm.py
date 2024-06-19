import os
import warnings
import csv
import pandas as pd
from langchain.docstore.document import Document 
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import CanineModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings('ignore')
os.environ["HF_HOME"] = "./cache/"

# Load data
df = pd.read_csv('./IM_RAG_Ans_data.csv')

# Metadata, Text splitter
dir_path = './IM_RAG_Ans_data.csv'
columns_to_embed = df.columns.tolist()
columns_to_metadata = df.columns.tolist()

docs = []
with open(dir_path, newline="") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
        to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
        to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    length_function=len
)
documents = splitter.split_documents(docs)

# Load DB
results = {}
model_name = 'google/canine-c'
model = CanineModel.from_pretrained(model_name)
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = Chroma(persist_directory=f"./documents_test_db", embedding_function=hf)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.92, 'k': 10}
)

# Prompt template
template = '''너는 영업직군으로 들어온 면접자에 대해 의사소통 역량을 평가하는 면접관이야. 그리고 question은 면접자의 답변이야. context는 DB에서 나온 상위 n개의 데이터는 3점에 해당하는 예시 우수 답변들이야. 

    DB를 참고해서 면접자가 한 답변이 상, 중, 하로 너가가 평가해줘.
    그리고 의사소통 역량은 총 4가지의 카테고리가 존재해. 1. 의사소통 능력 2. 정확한 기술활용 3. 의사소통 기술활용 4. 명료한 내용구성 
    지금부터는 각 카테고리 별 평가 기준에 대해 제시해줄게. 1점은 하에 해당하고, 2점은 중, 3점은 상에 해당해. 

    1. 효과적 의견교환
    1점 : 의사소통 절차를 적절히 사용하여 본인의 의견을 지속적으로 표현하고 전달할 수 있습니다. 그러나, 문제의 핵심에 근거하며 자신의 의견을 설명하거나 조직 안에서 원활한 의사소통이 일어날 수 있도록 프로세스를 개선하는 능력은 상대적으로 부족하다고 판단됩니다.
    2점 : 의사소통 절차를 적절히 사용하여 본인의 의견을 지속적으로 표현하고 전달할 수 있으며, 문제의 핵심에 근거하여 자신의 의견을 설명할 수 있습니다. 다만, 조직 안에서 원활한 의사소통이 일어날 수 있도록 프로세스를 개선하는 능력은 상대적으로 부족하다고 판단됩니다.
    3점 : 의사소통 절차를 적절히 사용하여 문제의 핵심에 근거한 본인의 의견을 지속적으로 표현하고 전달할 수 있습니다. 또한, 조직 안에서 원활한 의사소통이 일어날 수 있도록 프로세스를 개선하는 능력도 있다고 판단됩니다.

    2. 의사소통 기술활용
    1점 : 적절한 자료를 사용하여 상대방이 이해하기 편한 방식으로 의사소통을 하려는 모습을 보입니다. 다만, 표정이나 제스처 등 비언어적인 방식의 의사소통 수단을 적극적으로 활용하거나 조직의 의사소통 방식을 개선하여 원활한 소통 문화를 조성하는 능력은 상대적으로 부족하다고 판단됩니다.
    2점 : 적절한 자료와 표정이나 제스처 등의 비언어적인 방식을 적극적으로 활용하여 본인의 의도를 효과적이고 이해하기 쉬운 방식으로 전달합니다. 그러나, 조직의 의사소통 방식을 개선하여 원활한 소통 문화를 조성하는 능력은 상대적으로 부족하다고 판단됩니다.
    3점 : 적절한 자료와 표정이나 제스처 등의 비언어적인 방식을 적극적으로 활용하여 본인의 의도를 효과적이고 이해하기 쉬운 방식으로 전달합니다. 또한, 조직의 의사소통 방식을 개선하여 원활한 소통 문화를 조성하는 능력도 있다고 판단됩니다.

    3. 정확한 의사소통
    1점 : 간결하고 명확하게 본인의 의견을 전달할 수 있습니다. 그러나, 상황에 따라 적절한 의사소통 방식을 활용하거나 본인의 의견을 다른 사람들이 이해하였는지 확인하며 조직 내 정확한 의사소통 분위기를 조성하는 능력은 상대적으로 부족하다고 판단됩니다.
    2점 : 간결하고 명확하게 본인의 의견을 전달하며, 상황에 따라 적절한 의사소통 방식을 활용할 수 있습니다. 다만, 본인의 의견을 다른 사람들이 이해하였는지 확인하며 조직 내 정확한 의사소통 분위기를 조성하는 능력은 상대적으로 부족하다고 판단됩니다.
    3점 : 상황에 따라 적절한 의사소통 방식을 활용하며 간결하고 명확하게 본인의 의견을 전달합니다. 핵심 내용을 명확히 전달할 수 있으며 조직원들이 정확히 이해했는지 확인하며 조직 내 정확한 의사소통 분위기를 조성하는 능력도 지니고 있다고 판단됩니다.

    4. 명료한 내용구성
    1점 : 다양한 자료를 활용하여 자신의 의견을 정리한 후 제시합니다. 다만, 본인이 전달하고자 하는 바를 구조화하여 전달한 후, 여러 맥락에서도 정확히 전달되고 있는지 확인하는 능력은 상대적으로 부족하다고 판단됩니다.
    2점 : 충분한 자료와 논리에 근거하여 자신의 의견을 구조화하여 전달합니다. 그러나, 본인이 전달하고자 하는 바가 여러 맥락에서도 정확히 전달되고 있는지 확인하는 능력은 상대적으로 부족하다고 판단됩니다.
    3점 : 충분한 자료와 논리에 근거하여 자신의 의견을 구조화하여 전달할 수 있습니다. 더 나아가, 본인이 전달하고자 하는 바가 여러 맥락에서도 정확히 전달되고 있는지 확인하는 능력도 지니고 있다고 판단됩니다.

    1. 지금부터 상,중,하로 평가하고, 2. 왜 그렇게 평가했는지에 대한 이유를 자세하게 말해.
    3. 그리고, retriever에서 몇개의 답변이 나왔는지 개수를 말해줘 

    If you don't know the answer just say you don't know, don't make it up

    1. 평가 : 
    2. 이유 :
    3. 상위 n개 답변 개수 :
    
    상위 n개 답변 개수가 0일 경우, 3번 항목에 0개로 출력해. 그리고 그 답변은 '하'야. 평가 부분에 하로 평가해.

    할 말이 없다면, 즉, 빈 리스트로 내보낼거면 평가 부분에 '하'로 출력해
    '잘 모르겠습니다' 라는 답변이 나와도 평가부분에 '하'로 출력해

    해당 폼에 맞게 출력해줘
    {context}

    Answer: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# LLM setup
model = ChatOpenAI(api_key="input your openai key", temperature=0)

# RAG Chain setup
rag_chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


def rag_process(answer):
    return rag_chain.invoke(answer)


if __name__ == "__main__":
    # Chain execution example
    ans = '''팀 내에서 의견 교환을 효과적으로 이루기 위해 다양한 방법을 사용해 왔습니다. 몇 가지 효과적이었던 방법을 말씀드리자면
첫째, 정기적인 회의를 통해 팀원들이 각자의 의견을 공유할 수 있도록 했습니다. 특히 주간 회의나 스프린트 리뷰 회의에서 프로젝트 진행 상황과 문제점을 논의하며 다양한 의견을 나누었습니다.
둘째, 익명 피드백 도구를 활용해 팀원들이 솔직하게 의견을 낼 수 있도록 했습니다. 이를 통해 팀원들은 눈치 보지 않고 자신의 생각을 자유롭게 표현할 수 있었습니다.
셋째, 특정 주제나 문제에 대해 브레인스토밍 세션을 진행했습니다. 이 세션에서는 모든 아이디어가 존중되며, 비판 없이 다양한 의견을 모으는 데 중점을 두었습니다.
마지막으로, 긍정적이고 건설적인 피드백 문화를 조성했습니다. 정기적으로 피드백을 주고받음으로써 팀원들이 자신의 의견이 존중받고 있다는 느낌을 받을 수 있도록 했습니다.
이와 같은 방법들을 통해 팀 내에서 활발하고 효과적인 의견 교환을 이루어 나갈 수 있었습니다.
'''

    result = rag_chain.invoke(ans)
    print(f'{result}')
