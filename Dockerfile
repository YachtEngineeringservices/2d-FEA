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

# Copy application code (use .dockerignore to exclude unnecessary files)
COPY src/ /app/src/

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Create output directory
RUN mkdir -p /app/output

# Expose port
EXPOSE 8501

# Minimal health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD streamlit run src/web_app_clean.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
