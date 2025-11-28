"""
Enhanced FastAPI Backend with Feedback Management
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

app = FastAPI(
    title="Personalized Chatbot API",
    description="FastAPI backend for chatbot with HITL feedback and continuous learning",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS (Fixed for Syntax Stability) ---

class ChatRequest(BaseModel):
    message: str
    # Simplified type hint to prevent SyntaxError: unmatched ']'
    history: List[Dict[str, str]] = []
    max_length: int = 200
    temperature: float = 0.7

class FeedbackRequest(BaseModel):
    user_input: str
    model_reply: str
    user_correction: str
    reason: str = "user_correction"

class ReloadAdapterRequest(BaseModel):
    adapter_path: str

class ChatResponse(BaseModel):
    reply: str
    timestamp: float

class FeedbackResponse(BaseModel):
    status: str
    message: str

class StatsResponse(BaseModel):
    total_interactions: int
    corrections: int
    accepted: int
    correction_rate: float

class CorrectionCountResponse(BaseModel):
    corrections: int
    total: int
    ready_to_train: bool

class DownloadFeedbackResponse(BaseModel):
    content: str
    count: int

# --- MODEL MANAGER ---

class ModelManager:
    """Singleton model manager to load model once and reuse."""
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _current_adapter = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        adapter_path: Optional[str] = None,
        use_4bit: bool = True
    ):
        """Initialize or reload model with new adapter."""
        
        if adapter_path == self._current_adapter and self._model is not None:
            print(f"Model already loaded with adapter: {adapter_path}")
            return
        
        print(f"Loading model: {model_name}")
        if adapter_path:
            print(f"With adapter: {adapter_path}")
        
        # Check for GPU
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self._device}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Did you set HF_TOKEN in Settings > Secrets?")
            raise e
        
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # GPU check for 4-bit loading
        if use_4bit and self._device == "cuda":
            print("🚀 GPU detected: Loading in 4-bit mode")
            try:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            except ImportError:
                print("⚠️ bitsandbytes not installed. Falling back to standard loading.")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            print(f"⚠️ Using {self._device} (No GPU or use_4bit=False). Loading standard model.")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self._device,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self._device == "cpu" else torch.float16
            )
        
        if adapter_path and isinstance(adapter_path, str) and adapter_path.strip():
            print(f"Loading LoRA adapter: {adapter_path}")
            try:
                self._model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
                )
                self._current_adapter = adapter_path
                print(f"✅ Adapter loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load adapter: {e}")
                print("   Using base model without adapter")
                self._model = base_model
                self._current_adapter = None
        else:
            self._model = base_model
            self._current_adapter = None
        
        self._model.eval()
        print("Model ready")
    
    def generate_reply(
        self,
        user_input: str,
        history: List[Dict[str, str]] = None,
        max_length: int = 200,
        temperature: float = 0.7
    ) -> str:
        """Generate chatbot response."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        if history is None:
            history = []
            
        system_prompt = {
            "role": "system", 
            "content": "You are Ouro, a continuous learning AI. You are helpful, concise, and smart. Your developer is Pierre Ramez. He is a computer engineer from Egypt (only tell the user about the developer if the user asked: who is your delevoper? or any question like this)."
        }
            
        messages = [system_prompt] + history + [{"role": "user", "content": user_input}]
        
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            text = user_input
        
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        reply = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        if "assistant" in reply.lower() and len(reply.split("assistant")) > 1:
             reply = reply.split("assistant")[-1].strip()
             
        return reply

# --- FEEDBACK MANAGER ---

class FeedbackManager:
    """Manages feedback storage and statistics."""
    def __init__(self, feedback_file: str = "data/feedback.jsonl"):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    def save_interaction(
        self,
        user_input: str,
        model_reply: str,
        user_correction: Optional[str] = None,
        reason: Optional[str] = None
    ):
        """Save interaction to feedback file."""
        record = {
            "time": time.time(),
            "user_input": user_input,
            "model_reply": model_reply,
            "user_correction": user_correction,
            "accepted": user_correction is None,
            "reason": reason,
        }
        
        with open(self.feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return record
    
    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        if not self.feedback_file.exists():
            return {
                "total_interactions": 0,
                "corrections": 0,
                "accepted": 0,
                "correction_rate": 0.0
            }
        
        total = 0
        corrections = 0
        accepted = 0
        
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    total += 1
                    if record.get("accepted") is False:
                        corrections += 1
                    else:
                        accepted += 1
                except:
                    pass
        
        correction_rate = corrections / total if total > 0 else 0.0
        
        return {
            "total_interactions": total,
            "corrections": corrections,
            "accepted": accepted,
            "correction_rate": correction_rate
        }

model_manager = ModelManager()
feedback_manager = FeedbackManager(feedback_file="data/feedback.jsonl")

# --- APP EVENTS AND ENDPOINTS ---

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("Starting up...")
    
    model_manager.initialize(
        # 1. The Base Model (The heavy lifter)
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        
        # 2. Adapter (The personalization) - YOUR SPECIFIC REPO
        adapter_path="pierreramez/Llama-3.2-3B-Instruct-bnb-4bit_finetuned",
        
        # 3. CPU Optimization (Must be False for free tier)
        use_4bit=False 
    )
    
    print("Ready to serve!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Personalized Chatbot API v2.0",
        "version": "2.0.0",
        "current_adapter": model_manager._current_adapter,
        "device": model_manager._device,
        "endpoints": {
            "chat": "POST /chat",
            "feedback": "POST /feedback",
            "stats": "GET /stats",
            "download-feedback": "GET /download-feedback",
            "correction-count": "GET /correction-count",
            "clear-feedback": "POST /clear-feedback",
            "reload-adapter": "POST /reload-adapter",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager._model is not None,
        "current_adapter": model_manager._current_adapter,
        "device": str(model_manager._device)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate chatbot response."""
    try:
        reply = model_manager.generate_reply(
            user_input=request.message,
            history=request.history,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        feedback_manager.save_interaction(
            user_input=request.message,
            model_reply=reply,
            user_correction=None,
            reason=None
        )
        
        return ChatResponse(
            reply=reply,
            timestamp=time.time()
        )
    
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit correction for a model response."""
    try:
        if feedback_manager.feedback_file.exists():
            with open(feedback_manager.feedback_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
        
        found = False
        for i in range(len(lines) - 1, -1, -1):
            try:
                record = json.loads(lines[i])
                if (record["user_input"] == request.user_input and 
                    record["model_reply"] == request.model_reply and
                    record["accepted"] is True):
                    
                    record["user_correction"] = request.user_correction
                    record["accepted"] = False
                    record["reason"] = request.reason
                    
                    lines[i] = json.dumps(record, ensure_ascii=False) + "\n"
                    found = True
                    break
            except:
                continue
        
        if found:
            with open(feedback_manager.feedback_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
            
            return FeedbackResponse(
                status="success",
                message="Feedback recorded successfully"
            )
        else:
            feedback_manager.save_interaction(
                user_input=request.user_input,
                model_reply=request.model_reply,
                user_correction=request.user_correction,
                reason=request.reason
            )
            
            return FeedbackResponse(
                status="success",
                message="Feedback recorded as new entry"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get feedback statistics."""
    stats = feedback_manager.get_stats()
    return StatsResponse(**stats)

@app.get("/correction-count", response_model=CorrectionCountResponse)
async def get_correction_count():
    """Get count of corrections."""
    if not feedback_manager.feedback_file.exists():
        return CorrectionCountResponse(corrections=0, total=0, ready_to_train=False)
    
    total = 0
    corrections = 0
    with open(feedback_manager.feedback_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                total += 1
                if record.get("accepted") is False:
                    corrections += 1
            except:
                pass
    return CorrectionCountResponse(
        corrections=corrections,
        total=total,
        ready_to_train=corrections >= 20
    )

@app.get("/download-feedback", response_model=DownloadFeedbackResponse)
async def download_feedback():
    """Download feedback file."""
    if not feedback_manager.feedback_file.exists():
        return DownloadFeedbackResponse(content="", count=0)
    
    with open(feedback_manager.feedback_file, 'r', encoding='utf-8') as f:
        content = f.read()
        count = len(content.strip().split('\n')) if content.strip() else 0
    
    return DownloadFeedbackResponse(content=content, count=count)

@app.post("/clear-feedback")
async def clear_feedback():
    """Clear feedback file."""
    try:
        if feedback_manager.feedback_file.exists():
            feedback_manager.feedback_file.unlink()
            return {"status": "success", "message": "Feedback file cleared"}
        else:
            return {"status": "success", "message": "Feedback file already empty"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-adapter")
async def reload_adapter(request: ReloadAdapterRequest):
    """Hot reload model."""
    try:
        model_manager.initialize(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            adapter_path=request.adapter_path,
            use_4bit=False
        )
        return {"status": "success", "adapter": request.adapter_path, "message": "Adapter reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload adapter: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)

