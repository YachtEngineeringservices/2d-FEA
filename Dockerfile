# Optimized Dockerfile for Railway deployment
FROM dolfinx/dolfinx:v0.8.0

# Install only essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/*

# Install Python packages with dependencies but optimize for size
RUN pip3 install --no-cache-dir \
    streamlit==1.32.0 \
    matplotlib==3.8.3 \
    pandas==2.2.1 \
    plotly==5.19.0 \
    meshio==5.3.4 \
    h5py==3.10.0 \
    xarray==2024.2.0 \
    && pip3 cache purge

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
# Unset conflicting Streamlit environment variables\n\
unset STREAMLIT_SERVER_PORT\n\
unset STREAMLIT_SERVER_ADDRESS\n\
unset STREAMLIT_SERVER_HEADLESS\n\
PORT=${PORT:-8501}\n\
echo "Starting Streamlit on port $PORT"\n\
echo "Available environment variables:"\n\
env | grep -E "(PORT|STREAMLIT)" || true\n\
exec streamlit run src/web_app_clean.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables (remove conflicting Streamlit port settings)
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create output directory
RUN mkdir -p /app/output

# Expose port
EXPOSE 8501

# Minimal health check (use static port for health check)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD ["/app/start.sh"]
