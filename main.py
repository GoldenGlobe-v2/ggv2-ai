# main.py
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import requests
from pypdf import PdfReader
import io

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

        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"

            print("----추출된 텍스트----")
            print(extracted_text)
            print("-------------------")

            #TODO: 추출된 텍스트를 Vector DB에 저장하는 로직 추가

            return {"message" : "PDF 처리와 텍스트 추출이 성공적으로 완료됐습니다."}

    except Exception as e:
        print(f"PDF 처리 실패: {e}")
        return {"message" : f"PDF 처리 실패. Error: {e}"}


# --- /chat API 엔드포인트 ---
@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    print(f"Chat 요청 받음: travel_list_id={request.travel_list_id}, question='{request.question}'")
    # TODO: 여기에 Vector DB에서 관련 내용을 검색하고 LLM에게 답변을 생성하도록 요청하는 로직 추가
    return {"answer": f"'{request.question}'에 대한 임시 답변입니다."}