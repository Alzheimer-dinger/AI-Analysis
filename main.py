import os
import json
import asyncio
import aiohttp
import uuid
import datetime
from typing import Dict, Any, List
import traceback
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify

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

# 타임아웃 설정
API_TIMEOUT_SECONDS = 120  # Vertex AI 엔드포인트는 처리 시간이 더 길 수 있음
DB_TIMEOUT_SECONDS = 10

# Vertex AI 전역 초기화 (한 번만 실행)
if PROJECT_ID and LOCATION:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI initialized: project={PROJECT_ID}, location={LOCATION}")

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
            print("Warning: Response was truncated due to MAX_TOKENS")
            # 기본 응답 생성
            return '{"title": "대화 기록", "summary": "대화 내용이 기록되었습니다."}'
        
        if not candidate.content or not candidate.content.parts:
            print(f"Warning: Empty content, finish_reason: {candidate.finish_reason}")
            return '{"title": "대화 기록", "summary": "대화 분석을 완료했습니다."}'
        
        return response.text
        
    except ValueError as ve:
        if "MAX_TOKENS" in str(ve) or "safety filters" in str(ve):
            print(f"Gemini API blocked or truncated: {ve}")
            return '{"title": "대화 기록", "summary": "대화가 분석되었습니다."}'
        else:
            print(f"ValueError in generate_summary_sync: {ve}")
            raise
    except Exception as e:
        print(f"Error in generate_summary_sync: {e}")
        traceback.print_exc()
        raise

async def get_summary_and_title(conversation: List[Dict[str, str]]) -> Dict[str, str]:
    """LLM(Gemini)을 사용하여 대화의 제목과 요약을 생성합니다."""
    try:
        # 입력 검증
        if not conversation:
            print("Warning: Empty conversation provided")
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
            print(f"Conversation truncated from {original_length} to {max_chars} characters")
        
        print(f"Processing conversation with {len(conversation)} turns, {len(convo_text)} characters")
        
        # 비동기로 동기 함수 실행
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(
            executor,
            generate_summary_sync,
            convo_text
        )
        
        print(f"Raw response from Gemini: {response_text[:200]}...")  # 디버깅용
        
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
        
        print(f"Summary generated successfully: title='{title}', summary_length={len(summary)}")
        return {"title": title, "summary": summary}
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        if 'response_text' in locals():
            print(f"Failed to parse response: {response_text[:500]}")
        
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
        print(f"Unexpected error in get_summary_and_title: {e}")
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
        # 인증 토큰 획득
        creds, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        
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
                print(f"Unexpected label: {label}")
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
            
            print(f"Dementia analysis completed: label={label}, confidence={confidence}, risk_score={risk_score}")
            return {"risk_score": risk_score}

    except aiohttp.ClientError as e:
        print(f"HTTP error in get_dementia_analysis: {e}")
        return {"risk_score": -1.0}
    except Exception as e:
        print(f"Error in get_dementia_analysis: {e}")
        traceback.print_exc()
        return {"risk_score": -1.0}


async def get_emotion_analysis(session: aiohttp.ClientSession, conversation: List[Dict[str, str]]) -> Dict[str, float]:
    """Hugging Face Zero-Shot 모델로 감정을 분석합니다."""
    try:
        if not HF_TOKEN:
            print("Warning: HF_TOKEN not set, skipping emotion analysis")
            return {label: 0.0 for label in ["happy", "sad", "angry", "surprised", "bored"]}
        
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        
        # DB 스키마에 정의된 감정 레이블 사용
        candidate_labels = ["happy", "sad", "angry", "surprised", "bored"]
        
        # 환자의 대화 내용만 추출
        patient_utterances = [turn['content'] for turn in conversation if turn['speaker'] == 'patient']
        
        if not patient_utterances:
            print("No patient conversation found for emotion analysis.")
            return {label: 0.0 for label in candidate_labels}
        
        # 대화 내용 결합 (공백으로 구분)
        convo_text = " ".join(patient_utterances)
        
        # 텍스트 길이 제한 (Hugging Face API 제한 고려)
        max_length = 1000
        if len(convo_text) > max_length:
            convo_text = convo_text[:max_length]
            print(f"Patient conversation truncated to {max_length} characters for emotion analysis")

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
            
            print(f"Emotion analysis completed: {final_emotions}")
            return final_emotions

    except aiohttp.ClientError as e:
        print(f"HTTP error in get_emotion_analysis: {e}")
        return {label: -1.0 for label in ["happy", "sad", "angry", "surprised", "bored"]}
    except Exception as e:
        print(f"Error in get_emotion_analysis: {e}")
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
        print(f"Database pool created successfully: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        return pool
    except Exception as e:
        print(f"Failed to create database pool: {e}")
        raise


async def save_report(pool: aiomysql.Pool, session_id: str, content: Dict[str, str]) -> bool:
    """분석 리포트(제목, 요약)를 DB에 저장합니다."""
    try:
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                report_id = str(uuid.uuid4())
                created_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                # content를 JSON 문자열로 저장
                content_str = json.dumps(content, ensure_ascii=False)
                
                sql = "INSERT INTO reports (id, session_id, content, createdAt) VALUES (%s, %s, %s, %s)"
                await cursor.execute(sql, (report_id, session_id, content_str, created_at))
                await conn.commit()
                
                print(f"Report saved: {report_id} for session {session_id}")
                return True
                
    except Exception as e:
        print(f"Error saving report: {e}")
        traceback.print_exc()
        return False


async def save_dementia_analysis(pool: aiomysql.Pool, session_id: str, user_id: str, data: Dict[str, float]) -> bool:
    """치매 분석 결과를 DB에 저장합니다."""
    try:
        risk_score = data.get('risk_score', -1.0)
        
        # 오류 값(-1.0)인 경우 저장하지 않음
        if risk_score < 0:
            print(f"Skipping dementia analysis save due to error value: {risk_score}")
            return False
        
        async with pool.acquire() as conn:
            async with conn.cursor() as cursor:
                analysis_id = str(uuid.uuid4())
                created_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                sql = """
                    INSERT INTO dementia_analysis 
                    (id, session_id, user_id, risk_score, createdAt) 
                    VALUES (%s, %s, %s, %s, %s)
                """
                await cursor.execute(sql, (analysis_id, session_id, user_id, risk_score, created_at))
                await conn.commit()
                
                print(f"Dementia analysis saved: {analysis_id} for session {session_id}, risk_score={risk_score}")
                return True
                
    except Exception as e:
        print(f"Error saving dementia analysis: {e}")
        traceback.print_exc()
        return False


async def save_emotion_analysis(pool: aiomysql.Pool, session_id: str, user_id: str, data: Dict[str, float]) -> bool:
    """감정 분석 결과를 DB에 저장합니다."""
    try:
        # 모든 값이 오류 값(-1.0)인 경우 저장하지 않음
        if all(score < 0 for score in data.values()):
            print("Skipping emotion analysis save due to all error values")
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
                    (id, session_id, user_id, happy, sad, angry, surprised, bored, createdAt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    analysis_id, session_id, user_id,
                    happy, sad, angry, surprised, bored,
                    created_at
                )
                await cursor.execute(sql, values)
                await conn.commit()
                
                print(f"Emotion analysis saved: {analysis_id} for session {session_id}")
                return True
                
    except Exception as e:
        print(f"Error saving emotion analysis: {e}")
        traceback.print_exc()
        return False


# --- 메인 비동기 함수 ---

async def main(request_json: Dict[str, Any]) -> Dict[str, Any]:
    """메인 비동기 로직"""
    # 요청 데이터 추출 및 검증
    session_id = request_json.get("session_id")
    user_id = request_json.get("user_id")
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

    print(f"Processing session {session_id} for user {user_id}")
    print(f"Conversation has {len(conversation)} turns")
    print(f"Audio URL: {audio_url}")

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
    http_session = None
    
    try:
        # DB 커넥션 풀 생성
        db_pool = await create_db_pool()
        
        # HTTP 세션 생성
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=10)
        http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        
        # 세 가지 분석 작업을 병렬로 실행
        print("Starting parallel analysis tasks...")
        
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
        
        # 결과 확인 및 처리
        for idx, (result, name) in enumerate(zip(results, ['summary', 'dementia', 'emotion'])):
            if isinstance(result, Exception):
                print(f"Analysis failed - {name}: {result}")
                results_summary["analyses_failed"].append(name)
            else:
                print(f"Analysis completed - {name}")
                results_summary["analyses_completed"].append(name)

        # DB 저장 작업들
        print("Starting database save operations...")
        save_tasks = []
        
        if not isinstance(summary_result, Exception):
            save_tasks.append(('report', save_report(db_pool, session_id, summary_result)))
        
        if not isinstance(dementia_result, Exception):
            save_tasks.append(('dementia', save_dementia_analysis(db_pool, session_id, user_id, dementia_result)))
        
        if not isinstance(emotion_result, Exception):
            save_tasks.append(('emotion', save_emotion_analysis(db_pool, session_id, user_id, emotion_result)))

        if save_tasks:
            save_names = [name for name, _ in save_tasks]
            save_coroutines = [task for _, task in save_tasks]
            save_results = await asyncio.gather(*save_coroutines, return_exceptions=True)
            
            for name, result in zip(save_names, save_results):
                if isinstance(result, Exception):
                    print(f"DB save failed - {name}: {result}")
                    results_summary["db_saves_failed"].append(name)
                elif result is False:
                    print(f"DB save skipped - {name}")
                    results_summary["db_saves_failed"].append(name)
                else:
                    results_summary["db_saves_completed"].append(name)

        print(f"Processing completed for session {session_id}")
        return results_summary

    except Exception as e:
        print(f"Critical error in main: {e}")
        traceback.print_exc()
        raise
    
    finally:
        # 리소스 정리
        if http_session:
            await http_session.close()
        
        if db_pool:
            db_pool.close()
            await db_pool.wait_closed()


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
    print(f"=== Starting request {request_id} ===")
    
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return ({'error': 'Bad Request: No JSON payload', 'request_id': request_id}, 400)

        session_id = request_json.get('session_id', 'unknown')
        print(f"Processing request {request_id} for session {session_id}")

        # 비동기 메인 함수를 실행
        result = asyncio.run(main(request_json))
        
        # 성공 응답
        response = {
            'status': 'success',
            'request_id': request_id,
            'session_id': session_id,
            'summary': result
        }
        
        print(f"=== Request {request_id} completed successfully ===")
        return (response, 202)

    except ValueError as ve:
        error_msg = str(ve)
        print(f"Value Error in request {request_id}: {error_msg}")
        return ({
            'error': 'Bad Request',
            'message': error_msg,
            'request_id': request_id
        }, 400)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Internal Server Error in request {request_id}: {error_msg}")
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
