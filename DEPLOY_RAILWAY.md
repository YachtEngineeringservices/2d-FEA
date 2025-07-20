# Railway Deployment Guide

## Quick Deploy to Railway

1. **Fork/Clone this repository** to your GitHub account

2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app) and sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Connect your `2d-FEA` repository

3. **Configure the service**:
   - **Project Name**: `2d-fea-app` (or your preferred name)
   - **Branch**: `main`
   - Railway will automatically detect the Dockerfile

4. **Environment Variables** (Railway auto-detects most):
   - `PYTHONPATH`: `/app/src`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `PORT`: Railway provides this automatically

5. **Deploy**: Railway will automatically build and deploy

## Deployment Details

- **Plan**: Hobby plan ($5 minimum monthly usage) recommended
- **Free Trial**: 30 days free trial available
- **Build Time**: ~5-10 minutes (DOLFINx is a large image)
- **Memory**: ~1GB required for DOLFINx + Streamlit
- **Auto-scaling**: Railway handles scaling automatically

## Expected URL
Your app will be available at: `https://your-app-name.up.railway.app`

## Troubleshooting

- **Build fails**: Check that Dockerfile is in root directory
- **App doesn't start**: Verify Dockerfile CMD includes `${PORT:-8501}`
- **GMSH issues**: Railway has better scientific computing support than Render
- **Logs**: Use Railway dashboard to view build and runtime logs

## Features in Deployed App

✅ Full DOLFINx FEA solver  
✅ Interactive geometry input  
✅ Adaptive mesh generation  
✅ Real-time stress visualization  
✅ Zoom/pan controls  
✅ Professional results output  
✅ GMSH mesh generation (works reliably on Railway)

## Update Deployment

To update the deployed app:
1. Push changes to your GitHub repository
2. Railway will automatically rebuild and deploy

## Cost Structure

- **Free Trial**: 30 days free
- **Hobby Plan**: $5 minimum monthly usage
  - Up to 8GB RAM, 8 vCPU
  - Pay-per-use billing
- **Pro Plan**: $20 minimum monthly usage (if you need more resources)
  - Up to 32GB RAM, 32 vCPU

Railway's pay-per-use model means you only pay for actual usage, making it cost-effective for demonstration and development applications.

## Advantages over Other Platforms

- **Better Scientific Computing Support**: GMSH and DOLFINx work reliably
- **Docker-first Approach**: Superior container support
- **Pay-per-use**: No wasted money on unused resources
- **Auto-scaling**: Handles traffic spikes automatically
- **Simple Deployment**: Minimal configuration required
