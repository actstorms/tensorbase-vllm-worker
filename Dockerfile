# Use official vLLM image (already has vLLM, PyTorch, CUDA pre-installed)
FROM vllm/vllm-openai:v0.6.4.post1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install only the additional dependencies we need
RUN python3 -m pip install --no-cache-dir \
    httpx==0.28.1

# Copy server code
COPY server.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables
ENV MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.90
ENV TENSOR_PARALLEL_SIZE=1

# Reset entrypoint as the base image has one that tries to parse our CMD as arguments
ENTRYPOINT []

# Run the server
CMD ["python3", "server.py"]
