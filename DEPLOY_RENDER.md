# Render.com Deployment Guide

## Quick Deploy to Render.com

1. **Fork/Clone this repository** to your GitHub account

2. **Connect to Render.com**:
   - Go to [render.com](https://render.com) and sign in with GitHub
   - Click "New +" → "Web Service"
   - Connect your `2d-FEA` repository

3. **Configure the service**:
   - **Name**: `2d-fea-app` (or your preferred name)
   - **Branch**: `main`
   - **Build & Deploy**: Use `render.yaml` configuration (recommended)
   
   OR manually configure:
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Build Command**: Leave empty
   - **Start Command**: `streamlit run src/web_app_clean.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

4. **Environment Variables** (if not using render.yaml):
   - `PYTHONPATH`: `/app/src`
   - `STREAMLIT_SERVER_HEADLESS`: `true`

5. **Deploy**: Click "Create Web Service"

## Deployment Details

- **Plan**: Free tier (Starter) recommended for testing
- **Region**: Oregon (closest to US West Coast)
- **Health Check**: `/_stcore/health` (Streamlit's built-in endpoint)
- **Build Time**: ~5-10 minutes (DOLFINx is a large image)
- **Memory**: ~1GB required for DOLFINx + Streamlit

## Expected URL
Your app will be available at: `https://your-app-name.onrender.com`

## Troubleshooting

- **Build fails**: Check that Dockerfile is in root directory
- **App doesn't start**: Verify start command includes `$PORT` variable
- **Memory issues**: Consider upgrading from free tier if needed
- **Logs**: Use Render dashboard to view build and runtime logs

## Features in Deployed App

✅ Full DOLFINx FEA solver  
✅ Interactive geometry input  
✅ Adaptive mesh generation  
✅ Real-time stress visualization  
✅ Zoom/pan controls  
✅ Professional results output  

## Update Deployment

To update the deployed app:
1. Push changes to your GitHub repository
2. Render will automatically rebuild and deploy

## Cost

- **Free Tier**: Available with limitations (sleeps after 15min inactivity)
- **Starter Plan**: $7/month for always-on service
- **Standard Plan**: $25/month for more resources if needed

The free tier is sufficient for testing and light usage.
