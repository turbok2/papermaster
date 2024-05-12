import streamlit as st
from utils import print_messages, StreamHandler
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
import tempfile


# Assuming you have a chain setup
def log_step_output(step_result):
    print("Step output:", step_result)
    return step_result

# Suppose we want to use this vectorstore to find documents similar to a user's query
def get_similar_documents(query):
    # This example assumes you need to first convert the query into an embedding
    embedding = vectorstore.embedding_function.embed_documents([query])[0]
    similar_docs = vectorstore.similarity_search(embedding)
    return similar_docs
    
# Assuming we have a proper retriever method that gets the necessary context
def prepare_context(query):
    docs = get_similar_documents(query)
    context_text = ' '.join([doc['text'] for doc in docs])  # Simplified example
    return context_text
    
varRagDir="db11"
rag_flag = False
st.set_page_config(page_title="ChatGPT with Paper", page_icon="🦜")
st.title("🦜 ChatGPT with Paper  📝" )

# API KEY 설정
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 체팅 대화기록을 저장하는 store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

# RAG 구현
def save_pdf_to_db(uploaded_file):
    # varRagDir="db11"
    vectorstore = Chroma(persist_directory = varRagDir  , 
            embedding_function = 
            OpenAIEmbeddings()
            )
    print(vectorstore._collection.count())
    file_name = uploaded_file.name
    print(file_name)
    tmp_name = os.path.join('./tmp',file_name)
    result1 = vectorstore.get(where={"source": tmp_name})  # AI바우쳐 관련 내용
    print("삭제전 컬럼수 : " , len(result1))
    print("삭제전 리스트 건수: ", len(result1['ids']) )
    # 데이터 삭제 
    # https://python.langchain.com/docs/integrations/vectorstores/chroma#update-and-delete
    print("데이터 삭제 시작!!!!")
    for i in range(0, len(result1['ids']) ):
        print(i)
        iNum = result1['ids'][i]
        vectorstore._collection.delete(ids=iNum)
        
    # 파일을 임시 디렉토리에 저장
    # tmp_name = os.path.join('./tmp',file_name)
    with open(tmp_name, 'wb') as file:
        file.write(uploaded_file.getvalue())
    
    # 대상문서 RAG에 저장 
     # 임시 파일 경로를 사용하여 PyPDFLoader 초기화
    print(tmp_name)
    loader = PyPDFLoader(tmp_name)   
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)
    # print('all_splits[0]')
    # print(all_splits[0])
    # print('all_splits: ',len(all_splits))

    vectorstore = Chroma.from_documents(documents=all_splits, embedding = OpenAIEmbeddings(), 
            persist_directory=varRagDir)

    # retriever = vectorstore.as_retriever()
    
    #  저장된 문서가 있는지 검색
    result3 = vectorstore.get(where={"source": tmp_name})  # AI바우쳐 관련 내용
    print("저장후 컬럼수 : " , len(result3))
    print("저장후 리스트 건수: ", len(result3['ids']) )
    # print(result3)
    
    # 사용 후 임시 파일 삭제 (필요 시 주석 처리)
    os.remove(tmp_name)    
    
    # return retriever
    return vectorstore
    

with st.sidebar:
    #파일 업로드
    st.subheader('📝 논문 파일 업로드')
    uploaded_file = st.file_uploader('파일 선택')

    if uploaded_file is not None:
        vectorstore = save_pdf_to_db(uploaded_file)
        st.write('논문 파일 업로드 완료.')
        st.info(uploaded_file.name)
    else:
        st.write('논문 파일을 업로드하세요.')
        vectorstore = Chroma(persist_directory=varRagDir, embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    st.write('---')
    if st.checkbox('Using RAG'):
        rag_flag = True    
        st.info('읽어둔 논문 pdf 내용을 기반으로 답변합니다.')
    st.write('---')
    session_id =  st.text_input("Session_ID", value='abc123')

    clear_btn = st.button("대회기록 초기화")    
    if clear_btn:
        st.session_state["messages"] = []
        st.experimental_rerun()
# 이전 대화기록을 출력해주는 코드
print_messages()


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환



if user_input := st.chat_input("메시지를 입력해주세요."):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}") 
    # st.session_state["messages"].append(("user",user_input))
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    # LMM을 사용하여 AI의 답변을 생성

    # AI의 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # 1. 모델 생성
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler],
                         model_name="gpt-3.5-turbo", 
                        temperature=0)
#학습된 문서내에서 답변해 주세요. 학습된 문서내의 질문이 아니면 '질문하신 내용은 이 논문과 관련이 없습니다. '라고 대답해 주세요.
        if rag_flag:
            # 2. 프롬프트 생성
            template = """ 
            학습된 문서내에서 답변해 주세요.
            {context}
            Question: {question}
            Answer:"""        
            prompt = PromptTemplate.from_template(template)
            chain = ( #prompt | llm
                {"context":  RunnablePassthrough(), "question": RunnablePassthrough()} 
                # {"context":  retriever, "question": RunnablePassthrough()} 
                # | (prompt|log_step_output) # debug
                | prompt
                # |  (llm|log_step_output) # debug
                |  llm

            )
            # question="AI바우처가 뭐예요?"
            # result = chain.invoke(question).content   
            # print(result)
            docs = retriever.get_relevant_documents(user_input)
            print(len(docs))
            # print( docs[0])

            
            chain_with_memory = (
                RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
                    chain,  # 실행할 Runnable 객체
                    get_session_history,  # 세션 기록을 가져오는 함수
                    input_messages_key="question",  # 사용자 질문의 키
                    history_messages_key="history",  # 기록 메시지의 키
                )
            )
            # print('chain_with_memory: ')
            # print(chain_with_memory)
            
            response = chain_with_memory.invoke(
                # {"question": user_input},
                {"context":  docs, "question": user_input},
                # 세션 ID 설정
                config={"configurable": {"session_id": session_id}},
            )
            # print('response: ')
            # print(response)
            # print("Response from the model:", response.content)
        else:
            # 2. 프롬프트 생성
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "당신은 대학교수로 학생들의 논문지도를 50년 이상 해오고 있습니다. 대학원생들의 질문에 정확하고 상세하게 답변해 주세요.",
                    ),
                    # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),  # 사용자 질문을 입력
                ]
            )
            chain = prompt | llm  # 프롬프트와 모델을 연결하여 runnable  객체 생성
            chain_with_memory = (
                RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
                    chain,  # 실행할 Runnable 객체
                    get_session_history,  # 세션 기록을 가져오는 함수
                    input_messages_key="question",  # 사용자 질문의 키
                    history_messages_key="history",  # 기록 메시지의 키
                )
            )

            response = chain_with_memory.invoke(
                {"question": user_input},
                # 세션 ID 설정
                config={"configurable": {"session_id": session_id}},
            )

        msg = response.content    

        st.session_state["messages"].append(ChatMessage(role="assistant", content=response.content))

