"""
vLLM Worker for TensorBase Serverless

A production-ready vLLM worker exposing OpenAI-compatible endpoints
and integrating with the TensorBase serverless infrastructure.
"""

import os
import json
import asyncio
import time
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGING_FACE_HUB_TOKEN"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

# TensorBase integration
CALLBACK_URL = os.getenv("TENSORBASE_CALLBACK_URL", "")
WORKER_ID = os.getenv("TENSORBASE_WORKER_ID", "")
ENDPOINT_ID = os.getenv("TENSORBASE_ENDPOINT_ID", "")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vllm-worker")

# Global engine
engine: Optional[AsyncLLMEngine] = None

# ============================================
# Pydantic Models
# ============================================

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1

class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    echo: bool = False

class TensorBaseJobRequest(BaseModel):
    id: str
    input: Dict[str, Any]

# ============================================
# vLLM Engine Setup
# ============================================

async def initialize_engine():
    """Initialize the vLLM engine with the configured model."""
    global engine
    
    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info(f"Max model length: {MAX_MODEL_LEN}")
    logger.info(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    logger.info(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        enforce_eager=False,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("Model loaded successfully!")
    
    # Notify TensorBase that worker is ready
    await notify_worker_ready()

async def notify_worker_ready():
    """Notify TensorBase orchestrator that this worker is ready."""
    if not CALLBACK_URL or not WORKER_ID:
        logger.info("No callback URL configured, skipping ready notification")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CALLBACK_URL}/internal/worker/ready",
                json={
                    "workerId": WORKER_ID,
                    "endpointId": ENDPOINT_ID,
                },
                timeout=30.0
            )
            if response.status_code == 200:
                logger.info("Notified orchestrator: worker ready")
            else:
                logger.warning(f"Failed to notify ready: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to notify worker ready: {e}")

async def notify_job_complete(job_id: str, output: Dict[str, Any], error: Optional[str] = None):
    """Notify TensorBase orchestrator that a job is complete."""
    if not CALLBACK_URL:
        return
    
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "jobId": job_id,
                "workerId": WORKER_ID,
            }
            if error:
                payload["error"] = error
            else:
                payload["output"] = output
            
            await client.post(
                f"{CALLBACK_URL}/internal/job/complete",
                json=payload,
                timeout=30.0
            )
    except Exception as e:
        logger.error(f"Failed to notify job complete: {e}")

async def send_heartbeat():
    """Send periodic heartbeat to orchestrator."""
    if not CALLBACK_URL or not WORKER_ID:
        return
    
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{CALLBACK_URL}/internal/worker/heartbeat",
                    json={"workerId": WORKER_ID},
                    timeout=10.0
                )
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
        await asyncio.sleep(30)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for startup/shutdown."""
    # Startup
    await initialize_engine()
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat())
    
    yield
    
    # Shutdown
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="vLLM Worker",
    description="OpenAI-compatible vLLM inference server for TensorBase",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================
# Helper Functions
# ============================================

def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into a prompt string."""
    # Simple format - you may want to customize this for specific models
    formatted = ""
    for msg in messages:
        if msg.role == "system":
            formatted += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            formatted += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            formatted += f"Assistant: {msg.content}\n\n"
    formatted += "Assistant: "
    return formatted

async def generate_stream(prompt: str, params: SamplingParams, request_id: str) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    results_generator = engine.generate(prompt, params, request_id)
    
    previous_text = ""
    async for request_output in results_generator:
        for output in request_output.outputs:
            new_text = output.text[len(previous_text):]
            previous_text = output.text
            
            if new_text:
                chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
    
    # Final chunk
    final_chunk = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# ============================================
# API Endpoints
# ============================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "worker_id": WORKER_ID,
    }

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "vllm",
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = random_uuid()
    prompt = format_chat_prompt(request.messages)
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        n=request.n,
    )
    
    if request.stream:
        return StreamingResponse(
            generate_stream(prompt, sampling_params, request_id),
            media_type="text/event-stream"
        )
    
    # Non-streaming
    results = []
    async for output in engine.generate(prompt, sampling_params, request_id):
        results.append(output)
    
    final_output = results[-1]
    choices = []
    for i, output in enumerate(final_output.outputs):
        choices.append({
            "index": i,
            "message": {
                "role": "assistant",
                "content": output.text.strip()
            },
            "finish_reason": "stop"
        })
    
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": sum(len(c["message"]["content"].split()) for c in choices),
            "total_tokens": len(prompt.split()) + sum(len(c["message"]["content"].split()) for c in choices)
        }
    }

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible completions endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = random_uuid()
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        n=request.n,
    )
    
    results = []
    async for output in engine.generate(request.prompt, sampling_params, request_id):
        results.append(output)
    
    final_output = results[-1]
    choices = []
    for i, output in enumerate(final_output.outputs):
        text = output.text
        if request.echo:
            text = request.prompt + text
        choices.append({
            "index": i,
            "text": text,
            "finish_reason": "stop"
        })
    
    return {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": choices,
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": sum(len(c["text"].split()) for c in choices),
            "total_tokens": len(request.prompt.split()) + sum(len(c["text"].split()) for c in choices)
        }
    }

@app.post("/run")
async def run_job(request: TensorBaseJobRequest):
    """TensorBase-compatible job execution endpoint."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    job_id = request.id
    input_data = request.input
    
    try:
        # Extract parameters from input
        messages = input_data.get("messages", [])
        prompt = input_data.get("prompt")
        max_tokens = input_data.get("max_tokens", 256)
        temperature = input_data.get("temperature", 0.7)
        top_p = input_data.get("top_p", 1.0)
        stop = input_data.get("stop")
        
        # Determine the prompt
        if messages:
            # Chat format
            chat_messages = [ChatMessage(**m) for m in messages]
            final_prompt = format_chat_prompt(chat_messages)
        elif prompt:
            final_prompt = prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        
        request_id = random_uuid()
        
        results = []
        async for output in engine.generate(final_prompt, sampling_params, request_id):
            results.append(output)
        
        final_output = results[-1]
        generated_text = final_output.outputs[0].text.strip()
        
        output = {
            "text": generated_text,
            "usage": {
                "prompt_tokens": len(final_prompt.split()),
                "completion_tokens": len(generated_text.split()),
            }
        }
        
        # Notify completion
        await notify_job_complete(job_id, output)
        
        return {"status": "completed", "output": output}
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await notify_job_complete(job_id, {}, error=str(e))
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": str(e)}
        )

# ============================================
# Run with Uvicorn
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
