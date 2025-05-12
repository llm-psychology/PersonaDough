# backend.py
import logging
from pydantic import BaseModel
import aiofiles
from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import json
import os
from typing import List, Optional, Dict
import uvicorn
import glob

# Logging initialization
logging.basicConfig(
    level=logging.INFO,  # You can change to DEBUG or WARNING as needed
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

from module.QA_loader import QaLoader
from module.PERSONA_loader import PersonaLoader
from module.LLM_responder import (
    LLM_responder
)
from module.PERSONA_loader import (
    PersonaLoader
)
from module.QA_loader import (
    QaLoader
)
from interviewer.chat_with_persona import ( 
    ChatWith
)
from interviewer.interviewer_generator import (
    Interviewer,
    generate_persona_rag_by_id
)
from humanoid.humanoid_generator import (
    BaseInfoGenerator, 
    AttributeInjector, 
    StoryGenerator, 
    ToneGenerator, 
    SummarizedBehaviorGenerator, 
    AIParameterAnalyzer,
    CharacterGenerator,
    generate_a_persona
)

# =======================================================

# Database path
DATABASE_PATH = "humanoid/humanoid_database"
os.makedirs(DATABASE_PATH, exist_ok=True)
PHOTO_DIR = DATABASE_PATH + "/photo/"
os.makedirs(PHOTO_DIR, exist_ok=True)
character_generator = CharacterGenerator() # Initialize character generator

# New: conversation history database
CONVERSATION_HISTORY_PATH = "conversation_history"
os.makedirs(CONVERSATION_HISTORY_PATH, exist_ok=True)

# =======================================================

app = FastAPI(
    title="Humanoid Agent API",
    description="API for generating and managing humanoid agents, and persona interview simulator",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# frontend
@app.get("/")
def serve_frontend():
    return FileResponse("frontend.html", media_type="text/html")

# =======================================================

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class HumanoidResponse(BaseModel):
    id: str
    name: str
    file_path: str

class HumanoidFilter(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    gender: Optional[str] = None
    birthplace: Optional[str] = None
    traits: Optional[List[str]] = None
    social_abilities: Optional[List[str]] = None
    ability_attributes: Optional[List[str]] = None

class InterviewRequest(BaseModel):
    persona_id: str | None = None
    num_rounds: int = 10
    static_interview_data: str = ""  # User's input question
    session_id: Optional[str] = None  # New: session ID to maintain conversation history

class InterviewResponse(BaseModel):
    qa_pairs: List[Dict[str, str]]
    session_id: str  # Return the session ID for the client to use in subsequent requests

# New: Helper functions for conversation history management
async def load_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """Load conversation history for a given session"""
    history_file = os.path.join(CONVERSATION_HISTORY_PATH, f"{session_id}.json")
    if os.path.exists(history_file):
        async with aiofiles.open(history_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    return []

async def save_conversation_history(session_id: str, history: List[Dict[str, str]]):
    """Save conversation history for a given session"""
    history_file = os.path.join(CONVERSATION_HISTORY_PATH, f"{session_id}.json")
    async with aiofiles.open(history_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(history, ensure_ascii=False, indent=2))

# =======================================================

@app.get("/image/{id}")
async def get_image_by_id(id: str):
    image_path = os.path.join(PHOTO_DIR, f"{id}.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")

@app.post("/items")
async def create_item(item: Item):
    return item

@app.post("/interviews", response_model=InterviewResponse)
async def create_interview(request: InterviewRequest):
    """
    Handle interview requests with conversation history support.
    Uses session_id to maintain continuity between requests.
    """
    try:
        if not request.persona_id:
            raise HTTPException(status_code=400, detail="Persona ID is required")
            
        # Initialize chat handler
        chatbot = ChatWith()
        await chatbot.wait_until_ready()  # important
        
        # Get or create session ID
        session_id = request.session_id
        if not session_id:
            # Generate a new session ID if none provided
            import uuid
            session_id = f"{request.persona_id}_{str(uuid.uuid4())[:8]}"
            logger.info(f"Created new session: {session_id}")
            history = []
        else:
            # Load existing conversation history
            history = await load_conversation_history(session_id)
            logger.info(f"Loaded session {session_id} with {len(history)} previous exchanges")
        
        # Get user input from request
        user_question = request.static_interview_data
        logger.info(f"User input: {user_question}")
        
        # Load persona data and RAG database
        index, docs, qa_docs = await chatbot.interviewer_obj.load_rag_database(request.persona_id)
        
        # Format conversation history for context
        conversation_context = ""
        if history:
            conversation_context = "Previous conversation:\n" + "\n".join([
                f"User: {exchange['q']}\nPersona: {exchange['a']}" 
                for exchange in history[-5:]  # Include last 5 exchanges for context
            ])
        
        # Retrieve similar documents for context
        retrieved, distances = await chatbot.interviewer_obj.retrieve_similar_docs(user_question, index, docs)
        
        # If no similar documents with good confidence, set retrieved to None
        if len(distances) > 0 and all(dist < 0.5 for dist in distances):  # dist越小，離原話越近
            retrieved = None
        
        # Build prompt with persona data, conversation history, and context
        persona_data = chatbot.persona_loader_obj.personas[request.persona_id]
        
        # Build enhanced prompt that includes conversation history
        prompt = []
        system_content = f"""你是一個根據以下記憶資料模擬出來的人格：{persona_data['簡化行為']}\n
        記憶資料：
        {"\n".join(retrieved) if retrieved else "（無可參考的記憶資料）"}
        
        {conversation_context if conversation_context else ""}
        
        根據這些資料，用一致的口吻、用字遣詞、邏輯、口語化、對話模式來回答問題。你現在就是這個人，當有人問你是誰的時候，不能說你是AI助手，你要說出角色的名字。
        需要仔細衡量人格的背景，年齡和人生經歷須符合，回答的內容講求台灣的真實歷史性，不要產生不一致或不合邏輯的資料。
        仔細評估人格的語氣，情緒可以有點起伏，不要過度的正向，維持上下文的情緒。如果沒有上下文就維持平平的，無需過度開心或是憤怒。
        回應要保持連貫性，記住前面的對話內容，讓整個對話流程自然，減少反問使用者的機會。
        """
        
        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_question}
        ]
        
        # Generate AI response
        answer = await chatbot.interviewer_obj.simulate_persona_answer(prompt)
        logger.info(f"Response output: {answer}")
        
        # Add new exchange to history
        new_exchange = {"q": user_question, "a": answer}
        history.append(new_exchange)
        
        # Save updated history
        await save_conversation_history(session_id, history)
        
        # Format response
        qa_pairs = [new_exchange]  # Just return the latest exchange
        
        return InterviewResponse(qa_pairs=qa_pairs, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Interview API failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# New: endpoint to clear conversation history
@app.delete("/interviews/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear the conversation history for a given session"""
    history_file = os.path.join(CONVERSATION_HISTORY_PATH, f"{session_id}.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        return {"success": True, "message": f"Conversation history for session {session_id} cleared"}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

@app.get("/interviews/{session_id}")
async def get_conversation_history(session_id: str):
    """Get the full conversation history for a given session"""
    try:
        history = await load_conversation_history(session_id)
        return {"success": True, "session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@app.post("/generate", response_class=JSONResponse)
async def generate_humanoid(count: int = Query(1, gt=0, le=10, description="Number of humanoids to generate")):
    """Generate one or more new humanoid agents"""
    try:
        id = await generate_a_persona()
        await generate_persona_rag_by_id(id)
        return {"success": True, "count": 1, "humanoids": "please reflash the website"}
    
    except Exception as e:
        logger.error(f"Error generating humanoid: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating humanoid: {str(e)}")

@app.get("/humanoids", response_class=JSONResponse)
async def list_humanoids():
    '''
    @return
    {
    "success":true,"count":6,
    "humanoids":[
    {"id":"1955f47a6e","name":"王文欽","file_path":"humanoid/humanoid_database\\1955f47a6e_王文欽.json"},
    {"id":"278c94ca40","name":"陳秀蓮","file_path":"humanoid/humanoid_database\\278c94ca40_陳秀蓮.json"},
    {"id":"64ade2a584","name":"張麗卿","file_path":"humanoid/humanoid_database\\64ade2a584_張麗卿.json"},
    {"id":"d3e6670d22","name":"陳玉梅","file_path":"humanoid/humanoid_database\\d3e6670d22_陳玉梅.json"},
    {"id":"d4c097d155","name":"許文傑","file_path":"humanoid/humanoid_database\\d4c097d155_許文傑.json"},
    {"id":"f3ed0707e3","name":"曾麗華","file_path":"humanoid/humanoid_database\\f3ed0707e3_曾麗華.json"}]}
    '''
    try:
        files = glob.glob(os.path.join(DATABASE_PATH, "*.json"))
        humanoids = []
        
        for file_path in files:
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    humanoids.append({
                        "id": data["基本資料"]["id"],
                        "name": data["基本資料"]["姓名"],
                        "file_path": file_path
                    })
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue
        logger.info(f"Listed {len(humanoids)} humanoids.")
        return {"success": True, "count": len(humanoids), "humanoids": humanoids}
    
    except Exception as e:
        logger.error(f"Error listing humanoids: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing humanoids: {str(e)}")

@app.get("/humanoids/{humanoid_id}", response_class=JSONResponse)
async def get_humanoid(humanoid_id: str = Path(..., description="ID of the humanoid agent")):
    '''
    @return
    {
    "基本資料": {
        "id": "d4c097d155",
        "姓名": "許文傑",
        "年紀": 61,
        "性別": "男",
        "出生地": "花蓮"
    },
    "人格屬性": {
        "人格特質": [
        "足智多謀"
        ],
        "社交能力": [
        "關懷他人"
        ],
        "能力屬性": [
        "衝突解決",
        "生存技能"
        ]
    },
    "生平故事": "許文傑，生於花蓮，這個四面環山的地方孕育...",
    "語言行為": "許文傑是一位充滿智慧和關懷的律師...",
    "簡化行為": "許文傑，...",
    "AI參數": {
        "temperature": 0.7,
        "top_p": 0.8,
        "reason": "故事內容豐富且具有創意，需較高的多樣性與表現力。"
    }
    }
    '''
    try:
        # Find the file with the specified ID
        files = glob.glob(os.path.join(DATABASE_PATH, f"{humanoid_id}_*.json"))
        if not files:
            logger.warning(f"Humanoid with ID {humanoid_id} not found")
            raise HTTPException(status_code=404, detail=f"Humanoid with ID {humanoid_id} not found")
        
        file_path = files[0]
        
        # Read the file
        import aiofiles
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content) 
        logger.info(f"Fetched humanoid {humanoid_id}")
        return {"success": True, "humanoid": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting humanoid: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting humanoid: {str(e)}")

@app.post("/search", response_class=JSONResponse)
async def search_humanoids(filter_params: HumanoidFilter):
    """Search for humanoid agents using filters"""
    try:
        files = glob.glob(os.path.join(DATABASE_PATH, "*.json"))
        matching_humanoids = []
        
        for file_path in files:
            try:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Check if the humanoid matches all filters
                    matches = True
                    
                    # Age filter
                    if filter_params.age_min is not None and data["基本資料"]["年紀"] < filter_params.age_min:
                        matches = False
                    if filter_params.age_max is not None and data["基本資料"]["年紀"] > filter_params.age_max:
                        matches = False
                    
                    # Gender filter
                    if filter_params.gender and data["基本資料"]["性別"] != filter_params.gender:
                        matches = False
                    
                    # Birthplace filter
                    if filter_params.birthplace and filter_params.birthplace not in data["基本資料"]["出生地"]:
                        matches = False
                    
                    # Traits filter
                    if filter_params.traits:
                        if not all(trait in data["人格屬性"]["人格特質"] for trait in filter_params.traits):
                            matches = False
                    
                    # Social abilities filter
                    if filter_params.social_abilities:
                        if not all(ability in data["人格屬性"]["社交能力"] for ability in filter_params.social_abilities):
                            matches = False
                    
                    # Ability attributes filter
                    if filter_params.ability_attributes:
                        if not all(attr in data["人格屬性"]["能力屬性"] for attr in filter_params.ability_attributes):
                            matches = False
                    
                    if matches:
                        matching_humanoids.append({
                            "id": data["基本資料"]["id"],
                            "name": data["基本資料"]["姓名"],
                            "file_path": file_path,
                            "data": data
                        })
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue
        logger.info(f"Search result count: {len(matching_humanoids)}")
        return {
            "success": True, 
            "count": len(matching_humanoids), 
            "humanoids": matching_humanoids
        }
    
    except Exception as e:
        logger.error(f"Error searching humanoids: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error searching humanoids: {str(e)}")

# Start the server
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=20000, reload=True)