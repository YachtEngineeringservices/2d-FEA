#!/bin/bash

# Railway Deployment Script for 2D FEA Application

echo "🚂 Railway Deployment for 2D FEA Application"
echo "============================================="

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not in a git repository. Please run this from the project root."
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f Dockerfile ]; then
    echo "❌ Error: Dockerfile not found. Please ensure you're in the project root."
    exit 1
fi

echo "✅ Docker configuration found"

# Check git status
if [[ -n $(git status -s) ]]; then
    echo "⚠️  Warning: You have uncommitted changes. Commit them first for deployment."
    echo ""
    git status -s
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "🔍 Railway Deployment Instructions:"
echo ""
echo "1. Go to https://railway.app"
echo "2. Sign in with GitHub"
echo "3. Click 'New Project' → 'Deploy from GitHub repo'"
echo "4. Select your 2d-FEA repository"
echo "5. Railway will automatically:"
echo "   - Detect the Dockerfile"
echo "   - Set up build and deployment"
echo "   - Provide a deployment URL"
echo ""
echo "📋 Your repository is ready for Railway deployment!"
echo ""
echo "🔧 Configuration:"
echo "   - Dockerfile: ✅ Ready"
echo "   - Port: 8501 (automatically configured)"
echo "   - Environment: Docker with DOLFINx + GMSH"
echo ""
echo "💰 Pricing:"
echo "   - Free trial: 30 days"
echo "   - Hobby plan: $5 minimum usage"
echo "   - Pay-per-use billing"
echo ""
echo "🚀 After deployment, your app will be available at:"
echo "   https://your-app-name.up.railway.app"
echo ""

# Offer to open Railway website
read -p "Open Railway website now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v xdg-open > /dev/null; then
        xdg-open https://railway.app
    elif command -v open > /dev/null; then
        open https://railway.app
    else
        echo "Please open https://railway.app in your browser"
    fi
fi

echo ""
echo "📖 For detailed instructions, see DEPLOY_RAILWAY.md"
echo "✨ Happy deploying!"
