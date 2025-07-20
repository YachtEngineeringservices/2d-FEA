# Dockerfile for DOLFINx + Streamlit deployment on Render.com
FROM dolfinx/dolfinx:v0.8.0

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Streamlit and other Python packages (GMSH already included in base image)
RUN pip3 install --no-cache-dir \
    streamlit==1.32.0 \
    matplotlib==3.8.3 \
    pandas==2.2.1 \
    numpy==1.26.4 \
    scipy==1.12.0 \
    plotly==5.19.0 \
    meshio==5.3.4 \
    h5py==3.10.0 \
    xarray==2024.2.0

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY requirements_web.txt /app/

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Create output directory with proper permissions
RUN mkdir -p /app/output && chmod 755 /app/output

# Expose Streamlit port (Render.com uses PORT env var)
EXPOSE 8501

# Health check for Render.com
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the app
CMD streamlit run src/web_app_clean.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
