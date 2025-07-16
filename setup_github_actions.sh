#!/bin/bash
# GitHub Actions Setup Script for 2D FEA Windows Builds

echo "🚀 Setting up GitHub Actions for Windows Builds"
echo "=============================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit with 2D FEA application and GitHub Actions"
else
    echo "✅ Git repository already initialized"
fi

# Check GitHub Actions workflow
if [ -f ".github/workflows/build-windows.yml" ]; then
    echo "✅ GitHub Actions workflow is ready"
else
    echo "❌ GitHub Actions workflow missing - should have been created"
    exit 1
fi

# Add all files and commit
echo "📝 Adding files to git..."
git add .
git add .github/workflows/build-windows.yml
git add requirements_windows.txt
git add GITHUB_ACTIONS_SETUP.md

# Create commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Add GitHub Actions workflow for Windows builds

- Added comprehensive Windows build workflow
- Supports Python 3.11 and 3.12
- Creates distribution packages
- Automatic artifact uploads
- Release creation for tagged versions"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Create a GitHub repository at https://github.com/new"
echo "2. Copy the repository URL"
echo "3. Run these commands:"
echo ""
echo "   git remote add origin https://github.com/yourusername/2d-fea.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Go to your GitHub repository → Actions tab"
echo "5. The build will start automatically!"
echo ""
echo "📦 What you'll get:"
echo "   - 2D_FEA_Simple.exe"
echo "   - 2D_FEA_Torsion_Analysis.exe"
echo "   - Windows distribution packages"
echo "   - Automatic releases for tagged versions"
echo ""
echo "🏷️  To create a release:"
echo "   git tag v1.0.0"
echo "   git push origin v1.0.0"
echo ""
echo "✨ Your Windows executables will be built in the cloud!"

# Show current status
echo ""
echo "📊 Current Repository Status:"
echo "Files ready for GitHub:"
ls -la .github/workflows/
echo ""
echo "Requirements files:"
ls -la requirements*.txt
echo ""
echo "Documentation:"
ls -la *SETUP.md *GUIDE.md

echo ""
echo "🎉 Setup complete! Ready to push to GitHub."
