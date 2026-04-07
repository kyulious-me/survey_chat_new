import os
import httpx
import pdfplumber
import anthropic

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_DOMAIN = os.getenv("CONFLUENCE_DOMAIN", "jobis.atlassian.net")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.getenv(
    "PDF_PATH",
    os.path.join(_THIS_DIR, "onboarding.pdf")
)

CONFLUENCE_PAGE_IDS = [
    "2661122564",  # [TAX] 월세 OCR 기획
    "1430160045",  # [IIT X DP] 월세 서류검토 판단 조건
    "2763128883",  # 월세 관련 어드민 및 워크시트 관리 절차
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
frontend_dir = os.path.join(_THIS_DIR, "..")
app.mount("/img", StaticFiles(directory=os.path.join(frontend_dir, "img")), name="img")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.get("/chat")
async def serve_chat():
    return FileResponse(os.path.join(frontend_dir, "chat.html"))


# --- Document loading ---

def load_pdf_text(path: str) -> str:
    try:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[PDF 페이지 {i+1}]\n{page_text}")
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[PDF 로딩 실패: {e}]"


def strip_confluence_html(body: str) -> str:
    """Very lightweight tag stripper — avoids lxml/BeautifulSoup dependency."""
    import re
    # Replace common block tags with newlines
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</?(p|div|li|tr|th|td|h[1-6]|blockquote)[^>]*>", "\n", body, flags=re.IGNORECASE)
    # Remove all remaining tags
    body = re.sub(r"<[^>]+>", "", body)
    # Decode common HTML entities
    body = body.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    # Collapse excess blank lines
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


async def fetch_confluence_page(page_id: str) -> str:
    if not CONFLUENCE_EMAIL or not CONFLUENCE_API_TOKEN:
        return f"[Confluence 인증 정보 없음 — 페이지 {page_id} 건너뜀]"

    url = f"https://{CONFLUENCE_DOMAIN}/wiki/rest/api/content/{page_id}?expand=body.storage,title"
    auth = (CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, auth=auth)
            resp.raise_for_status()
            data = resp.json()
            title = data.get("title", f"페이지 {page_id}")
            raw_body = data["body"]["storage"]["value"]
            text = strip_confluence_html(raw_body)
            return f"=== Confluence: {title} ===\n{text}"
    except Exception as e:
        return f"[Confluence 페이지 {page_id} 로딩 실패: {e}]"


# In-memory document cache (loaded once at startup)
_document_context: str = ""


@app.on_event("startup")
async def load_documents():
    global _document_context
    parts = []

    # 1. PDF
    print("PDF 로딩 중...")
    pdf_text = load_pdf_text(PDF_PATH)
    parts.append(f"=== 추가공제 서류검토 PT Onboarding (PDF) ===\n{pdf_text}")
    print(f"PDF 로딩 완료 ({len(pdf_text):,} chars)")

    # 2. Confluence pages
    for page_id in CONFLUENCE_PAGE_IDS:
        print(f"Confluence 페이지 {page_id} 로딩 중...")
        page_text = await fetch_confluence_page(page_id)
        parts.append(page_text)
        print(f"  완료 ({len(page_text):,} chars)")

    _document_context = "\n\n" + "\n\n".join(parts)
    print(f"\n전체 문서 컨텍스트 준비 완료: {len(_document_context):,} chars")


# --- Chat API ---

SYSTEM_PROMPT = """당신은 삼쩜삼(자비스앤빌런즈)의 추가공제 서류검토 전문 어시스턴트입니다.
월세, 기부금, 중소기업취업감면(중취감) 관련 서류 심사를 담당하는 팀원들이 업무 중 발생하는 질문에 정확하고 실용적인 답변을 제공합니다.

아래는 내부 기획 문서와 가이드입니다. 이 문서를 기반으로 질문에 답변하세요.

{document_context}

---

답변 원칙:
1. 문서에 명확한 기준이 있으면 그대로 인용하며 답변하세요.
2. 문서에 없는 내용은 "문서에 명확한 기준이 없습니다"라고 솔직히 말하세요.
3. 실무에서 바로 적용할 수 있도록 구체적으로 답변하세요.
4. 반려/수정요청/입력완료 등 상태값 처리가 관련된 질문은 반드시 해당 상태값과 절차를 함께 안내하세요.
5. 한국어로 답변하세요."""


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
    if not _document_context:
        raise HTTPException(status_code=503, detail="문서 로딩 중입니다. 잠시 후 다시 시도해주세요.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build message history
    messages = []
    for h in req.history[-10:]:  # Keep last 10 turns for context
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    system = SYSTEM_PROMPT.format(document_context=_document_context)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        reply = response.content[0].text
        return ChatResponse(reply=reply)
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API 오류: {e}")


@app.get("/api/status")
async def status():
    return {
        "documents_loaded": bool(_document_context),
        "context_length": len(_document_context),
    }
