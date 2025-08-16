import os
import json
import asyncio
import aiohttp
import uuid
import datetime
import logging
from typing import Dict, Any, List, Optional
import traceback
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 환경변수 로딩 (로컬 실행 시)
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()


# Google Cloud Vertex AI (for Gemini and Dementia Analysis Endpoint)
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import google.auth
import google.auth.transport.requests

# Database Connector (async)
import aiomysql
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo

# --- 환경 변수 로드 (Cloud Function 환경 변수 또는 Secret Manager에 설정) ---
# Google Cloud
PROJECT_ID = os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_REGION", "us-central1")

# Vertex AI Endpoints
DEMENTIA_ANALYSIS_ENDPOINT_ID = os.environ.get("DEMENTIA_ANALYSIS_ENDPOINT_ID")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")

# Hugging Face Inference API
HF_API_URL = os.environ.get("HF_ZERO_SHOT_API_URL", "https://router.huggingface.co/hf-inference/models/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
HF_TOKEN = os.environ.get("HF_TOKEN")

# MySQL Database
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = int(os.environ.get("DB_PORT", 3306))

# MongoDB Database
MONGODB_URI = os.environ.get("MONGO_CONNECTION_STRING")
MONGODB_DB_NAME = os.environ.get("MONGO_DATABASE")

# 타임아웃 설정
API_TIMEOUT_SECONDS = 120  # Vertex AI 엔드포인트는 처리 시간이 더 길 수 있음
DB_TIMEOUT_SECONDS = 10

# Vertex AI 전역 초기화 (한 번만 실행)
def init_vertex_ai():
    """Vertex AI 초기화 - 환경변수 기반 credential 지원"""
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        from google.oauth2 import service_account
        credentials_info = json.loads(credentials_json)
        creds = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
    else:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
    logger.info(f"Vertex AI initialized: project={PROJECT_ID}, location={LOCATION}")

if PROJECT_ID and LOCATION:
    init_vertex_ai()

# ThreadPoolExecutor를 전역으로 생성 (재사용)
executor = ThreadPoolExecutor(max_workers=3)


# --- 비동기 분석 함수들 ---

def generate_summary_sync(conversation_text: str) -> str:
    """동기 함수로 Gemini API 호출"""
    try:
        model = GenerativeModel(model_name=GEMINI_MODEL_NAME)
        
        # 더 간단한 프롬프트 (토큰 수 절약)
        prompt = f"""대화를 분석해서 JSON으로 응답하세요:

{conversation_text}

형식:
{{"title": "제목 (10자 이내)", "summary": "요약 (50자 이내)"}}"""
        
        # Generation 설정 (적당한 토큰 수)
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,  # 적당한 temperature로 품질과 일관성 균형
            max_output_tokens=4096,
            top_p=0.9,
            top_k=40
        )
        
        # API 호출
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # 응답 처리 개선
        if not response.candidates:
            raise ValueError("No candidates in response")
        
        candidate = response.candidates[0]
        if candidate.finish_reason == "MAX_TOKENS":
            logger.warning("Response was truncated due to MAX_TOKENS")
            # 기본 응답 생성
            return '{"title": "대화 기록", "summary": "대화 내용이 기록되었습니다."}'
        
        if not candidate.content or not candidate.content.parts:
            logger.warning(f"Empty content, finish_reason: {candidate.finish_reason}")
            return '{"title": "대화 기록", "summary": "대화 분석을 완료했습니다."}'
        
        return response.text
        
    except ValueError as ve:
        if "MAX_TOKENS" in str(ve) or "safety filters" in str(ve):
            logger.warning(f"Gemini API blocked or truncated: {ve}")
            return '{"title": "대화 기록", "summary": "대화가 분석되었습니다."}'
        else:
            logger.error(f"ValueError in generate_summary_sync: {ve}")
            raise
    except Exception as e:
        logger.error(f"Error in generate_summary_sync: {e}")
        traceback.print_exc()
        raise

async def get_previous_comprehensive_report(
    pool: aiomysql.Pool,
    user_id: str
) -> Optional[Dict[str, Any]]:
    """특정 사용자의 가장 최근 종합 보고서를 DB에서 조회합니다."""
    sql = """
        SELECT 
            id, 
            base_report_id, 
            session_id, 
            user_id, 
            content, 
            created_at
        FROM reports
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """
    
    try:
        async with pool.acquire() as conn:
            # 결과를 딕셔너리 형태로 받기 위해 aiomysql.DictCursor 사용
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, (user_id,))
                result = await cursor.fetchone()
                
                if result:
                    logger.info(f"Found previous report for user {user_id}. Report ID: {result['id']}")
                    return result
                else:
                    logger.info(f"No previous report found for user {user_id}.")
                    return None

    except Exception as e:
        logger.error(f"Error fetching previous comprehensive report for user {user_id}: {e}")
        traceback.print_exc()
        return None
    
def generate_comprehensive_report_sync(
    summary_data: Dict[str, str],
    dementia_data: Dict[str, float],
    emotion_data: Dict[str, float],
    previous_report_data: Optional[Dict[str, Any]] # 이전 보고서 데이터 (없을 수 있음)
) -> str:
    """
    '오늘 하루 요약'과 '변화 추이'를 포함하는 마크다운 형식의 종합 보고서를 생성합니다.
    """
    try:
        # 1. 이전 보고서 데이터 처리
        if previous_report_data:
            previous_report_text = previous_report_data.get('content')
        else:
            previous_report_text = "이전 종합 보고서 기록이 없습니다. 첫 분석입니다."

        # 2. 금일 분석 결과 텍스트 처리
        dementia_risk = dementia_data.get('risk_score', -1.0)
        dementia_assessment = f"{dementia_risk:.1%}" if dementia_risk >= 0 else "분석 실패"
        
        # 감정 데이터에서 가장 점수가 높은 감정을 찾거나, 전반적인 상태를 요약할 수 있습니다.
        # 여기서는 간단히 모든 점수를 나열합니다.
        emotion_scores_str = ", ".join(
            [f"{k}: {v:.0%}" for k, v in emotion_data.items()]
        ) if all(v >= 0 for v in emotion_data.values()) else "0.0"

        # 프롬프트 구성
        prompt = f"""당신은 시니어의 건강 상태 변화를 추적하고, 이전 분석 결과와 오늘 분석 결과를 비교하여 핵심적인 변화와 추이를 설명하는 전문 AI 헬스케어 리포터입니다.

### 1. 이전 종합 보고서 요약
---
{previous_report_text}
---

### 2. 금일 대화 및 분석 결과
- **대화 내용 요약**: {summary_data.get('summary', 'N/A')}
- **인지 저하 위험도**: {dementia_assessment}
- **주요 감정**: {emotion_scores_str}
---

### [미션]
주어진 정보를 바탕으로, **'오늘 하루 요약'**과 **'이전 대비 변화 추이'** 두 부분으로 나누어 마크다운 보고서를 작성하세요. 아래 '출력 마크다운 형식 예시'의 구조를 반드시 따라주세요.

### [제약 조건]
- 한국어로 작성하세요.
- 전체 내용은 **1000자 이내**로 작성하세요.
- 별도의 인사나 부연 설명 없이 핵심 보고 내용만 바로 작성하세요.
- **PlainText 형식**으로 출력하세요.
"""

        # --- Gemini API 호출 ---
        model = GenerativeModel(model_name=GEMINI_MODEL_NAME)
        
        generation_config = GenerationConfig(
            temperature=0.7,
            max_output_tokens=2048
        )

        response = model.generate_content(prompt, generation_config=generation_config)
        
        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("Report content generation failed: Empty response from API.")
            
        return response.text

    except Exception as e:
        logger.error(f"Error in generate_comprehensive_report_sync: {e}")
        return None
    
async def get_summary_and_title(conversation: List[Dict[str, str]]) -> Dict[str, str]:
    """LLM(Gemini)을 사용하여 대화의 제목과 요약을 생성합니다."""
    try:
        # 입력 검증
        if not conversation:
            logger.warning("Empty conversation provided")
            return {
                "title": "대화 없음",
                "summary": "분석할 대화 내용이 없습니다."
            }
        
        # 대화 내용을 텍스트로 변환
        convo_text = "\n".join([
            f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}" 
            for turn in conversation
        ])
        
        # 대화 길이 제한
        max_chars = 8000  # 토큰 제한을 고려하여 줄임
        original_length = len(convo_text)
        if original_length > max_chars:
            convo_text = convo_text[:max_chars] + "\n... (대화 내용이 잘렸습니다)"
            logger.info(f"Conversation truncated from {original_length} to {max_chars} characters")
        
        logger.info(f"Processing conversation with {len(conversation)} turns, {len(convo_text)} characters")
        
        # 비동기로 동기 함수 실행
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            executor,
            generate_summary_sync,
            convo_text
        )
        
        logger.debug(f"Raw response from Gemini: {response_text[:200]}...")  # 디버깅용
        
        # 응답 정리
        response_text = response_text.strip()
        
        # 마크다운 코드블록 제거
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # JSON 파싱
        result = json.loads(response_text)
        
        # 결과 검증
        title = result.get("title", "").strip()
        summary = result.get("summary", "").strip()
        
        if not title:
            title = "대화 기록"
        if not summary:
            summary = "대화 내용이 분석되었습니다."
        
        # 길이 제한 적용
        if len(title) > 50:
            title = title[:50]
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        logger.info(f"Summary generated successfully: title='{title}', summary_length={len(summary)}")
        return {"title": title, "summary": summary}
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        if 'response_text' in locals():
            logger.error(f"Failed to parse response: {response_text[:500]}")
        
        # 기본값 반환 (대화 내용 기반)
        default_title = "대화 기록"
        if conversation:
            first_content = conversation[0].get('content', '')[:30]
            if first_content:
                default_title = first_content.split('.')[0][:10]
        
        return {
            "title": default_title,
            "summary": f"총 {len(conversation)}개의 대화가 기록되었습니다."
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in get_summary_and_title: {e}")
        traceback.print_exc()
        
        # 에러 타입별 메시지
        error_msg = "요약 생성 중 오류가 발생했습니다."
        if "PERMISSION_DENIED" in str(e):
            error_msg = "Vertex AI 권한 오류"
        elif "NOT_FOUND" in str(e):
            error_msg = "모델을 찾을 수 없음"
        elif "DEADLINE_EXCEEDED" in str(e):
            error_msg = "요청 시간 초과"
        
        return {
            "title": "대화 기록",
            "summary": error_msg
        }


async def get_dementia_analysis(session: aiohttp.ClientSession, audio_url: str) -> Dict[str, float]:
    """Vertex AI Endpoint로 치매 위험도를 분석하고 risk_score를 계산합니다."""
    try:
        # 인증 토큰 획득 (스코프 포함)
        credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            # 환경변수에서 JSON 자격증명 직접 사용
            import io
            from google.oauth2 import service_account
            credentials_info = json.loads(credentials_json)
            creds = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            project = credentials_info.get('project_id', PROJECT_ID)
        else:
            # 기존 방식 (파일 기반)
            creds, project = google.auth.default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        
        # 인증 정보 로깅 (디버깅용)
        logger.info(f"Authentication successful: project={project}, token_type={type(creds.token)}")
        if hasattr(creds, 'service_account_email'):
            logger.info(f"Service account: {creds.service_account_email}")
        else:
            logger.info("Using default credentials (not service account)")
        
        endpoint_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{DEMENTIA_ANALYSIS_ENDPOINT_ID}:predict"
        
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "instances": [{"gcs_uri": audio_url}]
        }

        timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
        async with session.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=timeout
        ) as response:
            response.raise_for_status()
            result = await response.json()

            # 응답에서 첫 번째 예측 결과를 추출
            predictions = result.get("predictions", [])
            if not predictions:
                raise ValueError("No predictions returned from the endpoint")
            
            prediction_result = predictions[0]
            label = prediction_result.get("prediction")
            confidence = float(prediction_result.get("confidence_percent", 0.0))
            
            # 라벨 검증
            if label not in ["SCI", "OTHERS"]:
                logger.warning(f"Unexpected label: {label}")
                return {"risk_score": -1.0}
            
            # Risk score 계산
            risk_score = 0.0
            if label == "SCI":
                # 치매 의심군일 경우, confidence를 그대로 risk_score로 사용
                risk_score = confidence / 100.0
            elif label == "OTHERS":
                # 대조군일 경우, 1에서 confidence를 뺀 값을 risk_score로 사용
                risk_score = 1.0 - (confidence / 100.0)
            
            # 범위 검증 (0.0 ~ 1.0)
            risk_score = max(0.0, min(1.0, risk_score))
            
            logger.info(f"Dementia analysis completed: label={label}, confidence={confidence}, risk_score={risk_score}")
            return {"risk_score": risk_score}

    except aiohttp.ClientError as e:
        logger.error(f"HTTP error in get_dementia_analysis: {e}")
        return {"risk_score": -1.0}
    except Exception as e:
        logger.error(f"Error in get_dementia_analysis: {e}")
        traceback.print_exc()
        return {"risk_score": -1.0}


async def get_emotion_analysis(session: aiohttp.ClientSession, conversation: List[Dict[str, str]]) -> Dict[str, float]:
    """Hugging Face Zero-Shot 모델로 감정을 분석합니다."""
    try:
        if not HF_TOKEN:
            logger.warning("HF_TOKEN not set, skipping emotion analysis")
            return {label: 0.0 for label in ["happy", "sad", "angry", "surprised", "bored"]}
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        # DB 스키마에 정의된 감정 레이블 사용
        candidate_labels = ["happy", "sad", "angry", "surprised", "bored"]
        
        # 환자의 대화 내용만 추출
        patient_utterances = [turn['content'] for turn in conversation if turn['speaker'] == 'patient']
        
        if not patient_utterances:
            logger.warning("No patient conversation found for emotion analysis.")
            return {label: 0.0 for label in candidate_labels}
        
        # 대화 내용 결합 (공백으로 구분)
        convo_text = " ".join(patient_utterances)
        
        # 텍스트 길이 제한 (Hugging Face API 제한 고려)
        max_length = 1000
        if len(convo_text) > max_length:
            convo_text = convo_text[:max_length]
            logger.info(f"Patient conversation truncated to {max_length} characters for emotion analysis")

        payload = {
            "inputs": convo_text,
            "parameters": {
                "candidate_labels": candidate_labels,
                "multi_label": False  # 단일 레이블 분류
            }
        }

        timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
        async with session.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        ) as response:
            response.raise_for_status()
            result = await response.json()
            
            # 결과 파싱
            # 응답 형식: {'sequence': '...', 'labels': ['sad', 'bored', ...], 'scores': [0.9, 0.05, ...]}
            if isinstance(result, list):
                result = result[0]  # 배치 처리 시 첫 번째 결과 사용
            
            labels = result.get('labels', [])
            scores = result.get('scores', [])
            
            if not labels or not scores:
                raise ValueError("Invalid response format from Hugging Face API")
            
            emotion_scores = dict(zip(labels, scores))
            
            # DB 스키마에 맞게 최종 결과 정리
            final_emotions = {}
            for label in candidate_labels:
                score = emotion_scores.get(label, 0.0)
                # 범위 검증 (0.0 ~ 1.0)
                final_emotions[label] = max(0.0, min(1.0, float(score)))
            
            logger.info(f"Emotion analysis completed: {final_emotions}")
            return final_emotions

    except aiohttp.ClientError as e:
        logger.error(f"HTTP error in get_emotion_analysis: {e}")
        return {label: -1.0 for label in ["happy", "sad", "angry", "surprised", "bored"]}
    except Exception as e:
        logger.error(f"Error in get_emotion_analysis: {e}")
        traceback.print_exc()
        return {label: -1.0 for label in ["happy", "sad", "angry", "surprised", "bored"]}

# --- 비동기 DB 저장 함수들 ---

async def create_db_pool() -> aiomysql.Pool:
    """DB 커넥션 풀 생성"""
    try:
        pool = await aiomysql.create_pool(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            autocommit=False,
            minsize=1,
            maxsize=5,  # Cloud Function의 동시 연결 제한 고려
            connect_timeout=DB_TIMEOUT_SECONDS,
            charset='utf8mb4'
        )
        logger.info(f"Database pool created successfully: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        return pool
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        raise

async def create_mongodb_client():
    """MongoDB 클라이언트 생성"""
    try:
        if not MONGODB_URI:
            raise ValueError("MONGO_CONNECTION_STRING environment variable is required")
        
        client = AsyncIOMotorClient(MONGODB_URI)
        # 연결 테스트
        await client.admin.command('ping')
        logger.info(f"MongoDB client created successfully: {MONGODB_DB_NAME}")
        return client
    except Exception as e:
        logger.error(f"Failed to create MongoDB client: {e}")
        raise


async def save_report(pool: aiomysql.Pool, session_id: str, user_id: str, content: Dict[str, str]) -> bool:
    """분석 리포트(제목, 요약)를 DB에 저장합니다."""
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                report_id = str(uuid.uuid4())
                created_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                # content를 JSON 문자열로 저장
                content_str = json.dumps(content, ensure_ascii=False)
                
                sql = "INSERT INTO reports (id, session_id, user_id, content, created_at) VALUES (%s, %s, %s, %s, %s)"
                await cursor.execute(sql, (report_id, session_id, user_id, content_str, created_at))
                await conn.commit()
                
                logger.info(f"Report saved: {report_id} for session {session_id}")
                return True
                
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        traceback.print_exc()
        return False

async def save_analysis_report_to_mongodb(
    mongodb_client, 
    document_id: str,
    title: str, 
    content: str
) -> bool:
    """MongoDB에 분석 리포트를 저장합니다. _id 기준으로 문서를 찾아서 title과 content 필드를 추가합니다."""
    try:
        from bson import ObjectId
        
        db = mongodb_client[MONGODB_DB_NAME]
        collection = db.conversation_sessions
        
        # ObjectId 변환 시도
        try:
            object_id = ObjectId(document_id)
            query_id = object_id
        except Exception:
            # ObjectId 변환 실패 시 문자열 그대로 사용
            query_id = document_id
            logger.info(f"Using document_id as string: {document_id}")
        
        # _id 기준으로 문서 업데이트
        result = await collection.update_one(
            {"_id": query_id},
            {
                "$set": {
                    "title": title,
                    "content": content,
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
            }
        )
        
        if result.matched_count > 0:
            logger.info(f"Analysis report added to MongoDB document: {document_id}")
            return True
        else:
            logger.warning(f"Document not found in MongoDB: {document_id}")
            return False
        
    except Exception as e:
        logger.error(f"Error saving analysis report to MongoDB: {e}")
        traceback.print_exc()
        return False


async def save_dementia_analysis(pool: aiomysql.Pool, session_id: str, user_id: str, data: Dict[str, float]) -> bool:
    """치매 분석 결과를 DB에 저장합니다."""
    try:
        risk_score = data.get('risk_score', -1.0)
        
        # 오류 값(-1.0)인 경우 저장하지 않음
        if risk_score < 0:
            logger.info(f"Skipping dementia analysis save due to error value: {risk_score}")
            return False
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                analysis_id = str(uuid.uuid4())
                created_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                sql = """
                    INSERT INTO dementia_analysis 
                    (id, session_id, user_id, risk_score, created_at) 
                    VALUES (%s, %s, %s, %s, %s)
                """
                await cursor.execute(sql, (analysis_id, session_id, user_id, risk_score, created_at))
                await conn.commit()
                
                logger.info(f"Dementia analysis saved: {analysis_id} for session {session_id}, risk_score={risk_score}")
                return True
                
    except Exception as e:
        logger.error(f"Error saving dementia analysis: {e}")
        traceback.print_exc()
        return False


async def save_emotion_analysis(pool: aiomysql.Pool, session_id: str, user_id: str, data: Dict[str, float]) -> bool:
    """감정 분석 결과를 DB에 저장합니다."""
    try:
        # 모든 값이 오류 값(-1.0)인 경우 저장하지 않음
        if all(score < 0 for score in data.values()):
            logger.info("Skipping emotion analysis save due to all error values")
            return False
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                analysis_id = str(uuid.uuid4())
                created_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                # 오류 값(-1.0)을 0.0으로 변환
                happy = max(0.0, data.get('happy', 0.0))
                sad = max(0.0, data.get('sad', 0.0))
                angry = max(0.0, data.get('angry', 0.0))
                surprised = max(0.0, data.get('surprised', 0.0))
                bored = max(0.0, data.get('bored', 0.0))
                
                sql = """
                    INSERT INTO emotion_analysis 
                    (id, session_id, user_id, happy, sad, angry, surprised, bored, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    analysis_id, session_id, user_id,
                    happy, sad, angry, surprised, bored,
                    created_at
                )
                await cursor.execute(sql, values)
                await conn.commit()
                
                logger.info(f"Emotion analysis saved: {analysis_id} for session {session_id}")
                return True
                
    except Exception as e:
        logger.error(f"Error saving emotion analysis: {e}")
        traceback.print_exc()
        return False

async def save_comprehensive_report(
    pool: aiomysql.Pool,
    session_id: str,
    user_id: str,
    previous_report_id: Optional[str],
    content: str,
) -> Optional[str]:
    """
    생성된 종합 보고서를 DB에 저장합니다.
    """
    # 1. 새로 저장할 리포트의 고유 ID 생성
    new_report_id = str(uuid.uuid4())
    
    # 2. 저장 시점의 UTC 시간을 ISO 8601 형식으로 생성
    created_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # 3. SQL 쿼리 준비 (INSERT)
    sql = """
        INSERT INTO reports 
        (id, base_report_id, session_id, user_id, content, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    # 4. 쿼리에 매핑할 값 준비
    # previous_report_id는 이전 리포트의 id이며, 없을 경우 NULL로 저장됩니다.
    values = (
        new_report_id,
        previous_report_id,
        session_id,
        user_id,
        content,
        created_at
    )

    try:
        # content 생성에 실패한 경우 저장하지 않음
        if content is None:
            logger.info("Skipping comprehensive report save due to all error values")
            return False
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:

                await cursor.execute(sql, values)
                await conn.commit()
                
                logger.info(f"Comprehensive report saved: {new_report_id} for session {session_id}")
                return True
                
    except Exception as e:
        logger.error(f"Error saving comprehensive report: {e}")
        traceback.print_exc()
        return False

# --- 메인 비동기 함수 ---

async def main(request_json: Dict[str, Any]) -> Dict[str, Any]:
    """메인 비동기 로직"""
    # 요청 데이터 추출 및 검증
    document_id = request_json.get("_id")  # MongoDB Document ID
    session_id = request_json.get("session_id")
    user_id = request_json.get("user_id")
    start_time = request_json.get("start_time")
    end_time = request_json.get("end_time")
    conversation = request_json.get("conversation")
    audio_url = request_json.get("audio_recording_url")

    if not session_id:
        raise ValueError("session_id is required")
    if not user_id:
        raise ValueError("user_id is required")
    if not conversation or not isinstance(conversation, list):
        raise ValueError("conversation must be a non-empty list")
    if not audio_url:
        raise ValueError("audio_recording_url is required")

    logger.info(f"Processing session {session_id} for user {user_id}")
    logger.info(f"Conversation has {len(conversation)} turns")
    logger.info(f"Audio URL: {audio_url}")

    # 결과 추적
    results_summary = {
        "session_id": session_id,
        "user_id": user_id,
        "analyses_completed": [],
        "analyses_failed": [],
        "db_saves_completed": [],
        "db_saves_failed": []
    }

    db_pool = None
    mongodb_client = None
    http_session = None
    
    try:
        # DB 커넥션 풀 생성
        db_pool = await create_db_pool()
        
        # MongoDB 클라이언트 생성
        if MONGODB_URI:
            mongodb_client = await create_mongodb_client()
        
        # HTTP 세션 생성
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=10)
        http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        
        # 세 가지 분석 작업을 병렬로 실행
        logger.info("Starting parallel analysis tasks...")
        
        summary_task = get_summary_and_title(conversation)
        dementia_task = get_dementia_analysis(http_session, audio_url)
        emotion_task = get_emotion_analysis(http_session, conversation)

        results = await asyncio.gather(
            summary_task,
            dementia_task,
            emotion_task,
            return_exceptions=True  # 오류가 발생해도 다른 작업은 계속 진행
        )

        summary_result, dementia_result, emotion_result = results
        comprehensive_report = None
        previous_report = None
        try:
            previous_report = await get_previous_comprehensive_report(db_pool, user_id)
            comprehensive_report = generate_comprehensive_report_sync(summary_result, dementia_result, emotion_result, previous_report)
        except Exception as e:
            logger.error(f'종합 보고서 생성 실패: {e}')
        
        # 결과 확인 및 처리
        for idx, (result, name) in enumerate(zip(results, ['summary', 'dementia', 'emotion'])):
            if isinstance(result, Exception):
                logger.error(f"Analysis failed - {name}: {result}")
                results_summary["analyses_failed"].append(name)
            else:
                logger.info(f"Analysis completed - {name}")
                results_summary["analyses_completed"].append(name)
        # 종합 보고서 결과 처리
        if isinstance(comprehensive_report, Exception) or comprehensive_report is None:
                logger.error(f"Analysis failed - comprehensive_report: {comprehensive_report}")
                results_summary["analyses_failed"].append('comprehensive_report')
        else:
            logger.info(f"Analysis completed - comprehensive_report")
            results_summary["analyses_completed"].append('comprehensive_report')

        # DB 저장 작업들
        logger.info("Starting database save operations...")
        save_tasks = []
        
        if not isinstance(summary_result, Exception):
            save_tasks.append(('report', save_report(db_pool, session_id, user_id, summary_result)))
        
        if not isinstance(dementia_result, Exception):
            save_tasks.append(('dementia', save_dementia_analysis(db_pool, session_id, user_id, dementia_result)))
        
        if not isinstance(emotion_result, Exception):
            save_tasks.append(('emotion', save_emotion_analysis(db_pool, session_id, user_id, emotion_result)))

        if not isinstance(comprehensive_report, Exception) and comprehensive_report is not None:
            save_tasks.append(('comprehensive_report', save_comprehensive_report(db_pool, session_id, user_id, 
                                                                                 previous_report.get("id") if previous_report is not None else None,
                                                                                 comprehensive_report)))

        if save_tasks:
            save_names = [name for name, _ in save_tasks]
            save_coroutines = [task for _, task in save_tasks]
            save_results = await asyncio.gather(*save_coroutines, return_exceptions=True)
            
            for name, result in zip(save_names, save_results):
                if isinstance(result, Exception):
                    logger.error(f"DB save failed - {name}: {result}")
                    results_summary["db_saves_failed"].append(name)
                elif result is False:
                    logger.warning(f"DB save skipped - {name}")
                    results_summary["db_saves_failed"].append(name)
                else:
                    results_summary["db_saves_completed"].append(name)

        # MongoDB에 분석 리포트 저장 (title과 content 추가)
        if mongodb_client and document_id and not isinstance(summary_result, Exception):
            try:
                title = summary_result.get('title', '대화 기록')
                content = summary_result.get('summary', '대화 내용이 분석되었습니다.')
                
                mongodb_save_result = await save_analysis_report_to_mongodb(
                    mongodb_client, 
                    document_id,
                    title, 
                    content
                )
                
                if mongodb_save_result:
                    results_summary["db_saves_completed"].append("mongodb_analysis_report")
                    logger.info(f"Analysis report saved to MongoDB document: {document_id}")
                else:
                    results_summary["db_saves_failed"].append("mongodb_analysis_report")
                    
            except Exception as e:
                logger.error(f"MongoDB analysis report save failed: {e}")
                results_summary["db_saves_failed"].append("mongodb_analysis_report")

        logger.info(f"Processing completed for session {session_id}")
        return results_summary

    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        traceback.print_exc()
        raise
    
    finally:
        # 리소스 정리
        if http_session:
            await http_session.close()
        
        if db_pool:
            db_pool.close()
            await db_pool.wait_closed()
            
        if mongodb_client:
            mongodb_client.close()


# --- Cloud Function Entry Point ---

def analyze_conversation_session(request):
    """HTTP Cloud Function의 진입점"""
    # CORS 처리 (필요한 경우)
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    if request.method != 'POST':
        return ({'error': 'Method Not Allowed', 'allowed_methods': ['POST']}, 405)
    
    request_id = str(uuid.uuid4())
    logger.info(f"=== Starting request {request_id} ===")
    
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return ({'error': 'Bad Request: No JSON payload', 'request_id': request_id}, 400)

        session_id = request_json.get('session_id', 'unknown')
        logger.info(f"Processing request {request_id} for session {session_id}")

        # 비동기 메인 함수를 실행
        result = asyncio.run(main(request_json))
        
        # 성공 응답
        response = {
            'status': 'success',
            'request_id': request_id,
            'session_id': session_id,
            'summary': result
        }
        
        logger.info(f"=== Request {request_id} completed successfully ===")
        return (response, 202)

    except ValueError as ve:
        error_msg = str(ve)
        logger.error(f"Value Error in request {request_id}: {error_msg}")
        return ({
            'error': 'Bad Request',
            'message': error_msg,
            'request_id': request_id
        }, 400)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Internal Server Error in request {request_id}: {error_msg}")
        traceback.print_exc()
        return ({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'request_id': request_id
        }, 500)

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    # Flask request를 Cloud Function 진입점에 전달
    result, status = analyze_conversation_session(request)
    return jsonify(result), status

if __name__ == "__main__":
    # 로컬 테스트용
    app.run(host="0.0.0.0", port=8080)
