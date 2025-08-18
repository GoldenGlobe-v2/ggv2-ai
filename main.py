# main.py
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import requests
from pypdf import PdfReader
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

load_dotenv()

# --- Pydantic 모델 정의 ---
# /index API가 받을 요청 Body의 구조
class IndexRequest(BaseModel):
    travel_list_id: int
    pdf_url: HttpUrl

# /chat API가 받을 요청 Body의 구조
class ChatRequest(BaseModel):
    travel_list_id: int
    question: str


# --- FastAPI 앱 생성 ---
app = FastAPI()

# 텍스트를 벡터로 변환하는 모델
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
llm = ChatGoogleGenerativeAI(embeddings="gemini-pro")
vector_stores = {}

@app.get("/")
def read_root():
    return {"message": "GoldenGlobe AI Server is running!"}


# --- /index API 엔드포인트 생성 ---
@app.post("/index")
def index_pdf(request: IndexRequest):
    try:
        response = requests.get(str(request.pdf_url))
        response.raise_for_status()

        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)

        extracted_text = "".join(page.extract_text() + "\n" for page in reader.pages)

        #AI 학습 로직 추가
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_text(extracted_text)

        db = FAISS.from_texts(docs, embeddings)

        travel_id = request.travel_list_id
        if travel_id not in vector_stores:
            vector_stores[travel_id] = db
        else:
            vector_stores[travel_id].merge_from(db)
        return {"message": "PDF 처리와 인덱싱이 성공적으로 완료되었습니다."}

    except Exception as e:
        return {"message" : f"PDF 처리 실패. Error: {e}"}


# --- /chat API 엔드포인트 ---
@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    travel_id = request.travel_list_id
    question = request.question

    # 해당 여행에 대해 학습된 벡터 저장소가 있는지 확인
    if travel_id not in vector_stores:
        return {"answer":"아직 학습된 PDF 문서가 없습니다. 먼저 문서를 업로드해주세요."}

    db = vector_stores[travel_id]

    # Langchain을 사용하여 QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    # QA 체인을 실행하여 답변 생성
    try:
        answer = qa_chain.run(question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": "답변을 생성하는 중 오류가 발생했습니다."}