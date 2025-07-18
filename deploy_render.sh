#!/bin/bash
# Deploy 2D FEA to Render.com

echo "🚀 Deploying 2D FEA with DOLFINx to Render.com"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "render.yaml" ]; then
    echo "❌ Error: render.yaml not found. Please run this script from the project root."
    exit 1
fi

# Check if changes are committed
if ! git diff --quiet; then
    echo "⚠️  Warning: You have uncommitted changes."
    echo "🔄 Committing changes before deployment..."
    git add .
    git commit -m "🚀 Deploy to Render.com - $(date)"
fi

# Push to main branch
echo "📤 Pushing to main branch..."
git push origin main

echo ""
echo "✅ Code pushed to GitHub!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://render.com"
echo "2. Create new Web Service"
echo "3. Connect your GitHub repository"
echo "4. Render will automatically use the render.yaml configuration"
echo "5. Wait ~10-15 minutes for Docker build to complete"
echo "6. Your app will be available at: https://2d-fea-dolfinx.onrender.com"
echo ""
echo "💰 Cost: $7/month (Starter plan)"
echo "🔧 Features: Full DOLFINx FEA, GMSH meshing, professional visualization"
echo "📊 Performance: 512MB RAM, 0.1 CPU, always-on service"
echo ""
echo "🎯 Your professional FEA analysis will be available worldwide!"
echo "🌐 Same accuracy as desktop version, accessible from any browser"
echo ""
echo "📚 For detailed instructions, see RENDER_DEPLOYMENT.md"
