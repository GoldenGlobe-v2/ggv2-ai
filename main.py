# main.py
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import requests
from pypdf import PdfReader
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# --- Pydantic 모델 정의 ---
class IndexRequest(BaseModel):
    travel_list_id: int
    pdf_url: HttpUrl

class ChatRequest(BaseModel):
    travel_list_id: int
    question: str

# --- FastAPI 앱 및 AI 모델 초기화 ---
app = FastAPI()

embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# embeddings="gemini-pro"가 아니라 model="gemini-pro"가 올바른 파라미터입니다.
llm = ChatGoogleGenerativeAI(model="gemini-pro")

vector_stores = {}

@app.get("/")
def read_root():
    return {"message": "GoldenGlobe AI Server is running!"}

@app.post("/index")
def index_pdf(request: IndexRequest):
    try:
        response = requests.get(str(request.pdf_url))
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        extracted_text = "".join(page.extract_text() + "\n" for page in reader.pages)

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

@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    travel_id = request.travel_list_id
    question = request.question

    if travel_id not in vector_stores:
        return {"answer": "아직 학습된 PDF 문서가 없습니다. 먼저 문서를 업로드해주세요."}

    db = vector_stores[travel_id]
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    try:
        answer = qa_chain.run(question)
        return {"answer": answer}
    except Exception as e:
        return {"answer": "답변을 생성하는 중 오류가 발생했습니다."}