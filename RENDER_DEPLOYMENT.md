# Render.com Deployment Guide

## 🚀 Deploy 2D FEA with Full DOLFINx Support on Render.com

This guide walks you through deploying your 2D FEA application with complete DOLFINx support on Render.com.

### ✅ Why Render.com?

- **Full DOLFINx support** via Docker
- **Professional FEA capabilities** (same as desktop)
- **Automatic deployments** from GitHub
- **Cost-effective** ($7/month for starter plan)
- **Easy setup** with minimal configuration
- **Always-on** service (no cold starts like serverless)

## 🛠️ Deployment Steps

### 1. **Sign Up for Render.com**
1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account
3. Authorize Render to access your repositories

### 2. **Create New Web Service**
1. Click **"New"** → **"Web Service"**
2. Connect your GitHub repository: `YachtEngineeringservices/2d-FEA`
3. Render will automatically detect the `render.yaml` configuration

### 3. **Configure Service Settings**
The `render.yaml` file automatically configures:
- **Service name**: `2d-fea-dolfinx`
- **Environment**: Docker
- **Plan**: Starter ($7/month)
- **Port**: 8501
- **Auto-deploy**: Enabled from main branch

### 4. **Deploy**
1. Click **"Create Web Service"**
2. Render will start building your Docker image
3. Build process takes ~10-15 minutes (includes DOLFINx compilation)
4. Once complete, you'll get a live URL like: `https://2d-fea-dolfinx.onrender.com`

## 📋 Service Configuration

### **Current Settings** (in `render.yaml`)
```yaml
services:
  - type: web
    name: 2d-fea-dolfinx
    env: docker
    dockerfilePath: ./Dockerfile
    plan: starter  # $7/month
    port: 8501
    envVars:
      - key: PYTHONPATH
        value: /app/src
      - key: STREAMLIT_SERVER_HEADLESS
        value: "true"
    healthCheckPath: /_stcore/health
    autoDeploy: true
    branch: main
```

### **Environment Variables**
- `PYTHONPATH`: Ensures FEA modules are found
- `STREAMLIT_SERVER_HEADLESS`: Optimizes for server deployment
- `ENVIRONMENT`: Set to production

### **Health Check**
- **Path**: `/_stcore/health`
- **Ensures**: Service is healthy and responding

## 💰 Cost Breakdown

### **Render.com Pricing**
- **Starter Plan**: $7/month
  - 512MB RAM
  - 0.1 CPU
  - Custom domain support
  - Always-on (no cold starts)
  - Automatic SSL

### **Usage Estimates**
- **Light usage**: Starter plan sufficient
- **Heavy usage**: Upgrade to Standard ($25/month) for better performance
- **Enterprise**: Pro plan ($85/month) for high availability

## 🔧 Docker Configuration

### **Base Image**
```dockerfile
FROM dolfinx/dolfinx:v0.8.0
```
- Official DOLFINx image with all dependencies
- Includes PETSc, SLEPC, MPI, and other FEA libraries

### **Python Dependencies**
```dockerfile
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
```

### **Optimizations**
- **Multi-stage build**: Reduces final image size
- **Health checks**: Ensures service reliability
- **Proper permissions**: Handles file system access
- **Environment variables**: Configures Streamlit for production

## 🚀 Post-Deployment

### **Accessing Your App**
1. **URL**: `https://2d-fea-dolfinx.onrender.com`
2. **Status**: Check deployment status in Render dashboard
3. **Logs**: View real-time logs for debugging

### **Features Available**
- ✅ **Full DOLFINx FEA solver**
- ✅ **GMSH mesh generation**
- ✅ **Professional stress field visualization**
- ✅ **Same accuracy as desktop version**
- ✅ **Multi-point geometry input**
- ✅ **Real-time progress feedback**
- ✅ **Automatic scaling**

### **Performance Expectations**
- **Startup time**: ~30 seconds (first load)
- **Mesh generation**: 5-30 seconds (depending on complexity)
- **FEA solving**: 10-60 seconds (depending on mesh size)
- **Visualization**: Near-instant

## 🔧 Troubleshooting

### **Common Issues**

#### **Build Failures**
```
Error: Failed to build Docker image
```
**Solutions:**
1. Check Dockerfile syntax
2. Verify base image availability
3. Review build logs in Render dashboard

#### **Service Not Starting**
```
Error: Service failed to start
```
**Solutions:**
1. Check health check endpoint
2. Verify port configuration (8501)
3. Review application logs

#### **DOLFINx Import Errors**
```
Error: No module named 'dolfinx'
```
**Solutions:**
1. Verify base image is `dolfinx/dolfinx:v0.8.0`
2. Check PYTHONPATH environment variable
3. Rebuild service

### **Performance Issues**
- **Upgrade plan**: Starter → Standard for better performance
- **Check logs**: Look for memory/CPU bottlenecks
- **Optimize mesh size**: Reduce default mesh size for faster computation

## 🔄 Continuous Deployment

### **Automatic Updates**
- **Push to main branch** → Automatic deployment
- **Build process** → ~10-15 minutes
- **Zero-downtime** → Gradual rollout

### **Manual Deployment**
1. Go to Render dashboard
2. Click **"Deploy latest commit"**
3. Wait for build completion

## 📊 Monitoring

### **Available Metrics**
- **Response time**
- **Memory usage**
- **CPU usage**
- **Error rates**
- **Deployment history**

### **Alerts**
- **Email notifications** for deployment failures
- **Slack integration** available
- **Custom webhooks** for monitoring

## 🎯 Next Steps

1. **Deploy the service** following the steps above
2. **Test full FEA functionality** with sample geometries
3. **Set up custom domain** (optional)
4. **Monitor performance** and upgrade plan if needed
5. **Share the URL** with users

## 📚 Additional Resources

- [Render.com Documentation](https://render.com/docs)
- [DOLFINx Documentation](https://docs.fenicsx.org)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)

## 🆘 Support

If you encounter issues:
1. Check the [Render.com status page](https://status.render.com)
2. Review application logs in Render dashboard
3. Check GitHub repository issues
4. Contact Render support if needed

---

**Ready to deploy professional FEA analysis to the cloud!** 🚀
