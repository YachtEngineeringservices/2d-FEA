# GitHub Actions Windows Build Setup

This repository is configured to automatically build Windows executables using GitHub Actions.

## 🚀 How It Works

### Automatic Builds
The GitHub Actions workflow automatically triggers on:
- **Push to main/master/develop branches**
- **Pull requests**
- **Git tags** (creates releases)
- **Manual trigger** (workflow_dispatch)

### Build Matrix
Builds are created for:
- **Python versions**: 3.11, 3.12
- **Architecture**: x64 (Windows 64-bit)

## 📦 Build Output

### Generated Executables:
- `2D_FEA_Simple.exe` - Main 2D FEA application
- `2D_FEA_Torsion_Analysis.exe` - Torsion analysis tool

### Distribution Packages:
- `2D_FEA_Windows_3.11_x64.zip` - Python 3.11 build
- `2D_FEA_Windows_3.12_x64.zip` - Python 3.12 build

## 🔧 Setup Instructions

### 1. Repository Setup
```bash
# Initialize git repository if not already done
git init
git add .
git commit -m "Initial commit with GitHub Actions"

# Add remote repository (replace with your GitHub repo)
git remote add origin https://github.com/YachtEngineeringservices/2d-FEA.git
git push -u origin main
```

### 2. GitHub Repository Settings
1. Go to your GitHub repository
2. Navigate to **Settings** → **Actions** → **General**
3. Ensure **Actions permissions** are enabled
4. Set **Workflow permissions** to "Read and write permissions"

### 3. Trigger Your First Build

#### Option A: Push to trigger build
```bash
git add .
git commit -m "Add GitHub Actions workflow"
git push origin main
```

#### Option B: Create a release tag
```bash
git tag v1.0.0
git push origin v1.0.0
```

#### Option C: Manual trigger
1. Go to **Actions** tab in GitHub
2. Select "Build Windows Executables"
3. Click "Run workflow"

## 📋 Build Process

### What the workflow does:
1. **🔽 Checkout**: Downloads your code
2. **🐍 Setup Python**: Installs Python 3.11/3.12
3. **🔧 Install Dependencies**: Visual C++ tools, Python packages
4. **📦 Install Packages**: numpy, matplotlib, PySide6, PyInstaller
5. **📝 Create Specs**: Generates PyInstaller configuration
6. **🔨 Build Apps**: Creates Windows executables
7. **🧪 Test**: Validates executables work
8. **📦 Package**: Creates ZIP distributions
9. **📤 Upload**: Saves artifacts for download

### Build time: ~15-20 minutes per build

## 📥 Downloading Built Applications

### From GitHub Actions:
1. Go to **Actions** tab
2. Click on a completed workflow run
3. Scroll to **Artifacts** section
4. Download the ZIP files

### From Releases (for tagged versions):
1. Go to **Releases** tab
2. Download executables from the latest release

## 🔍 Monitoring Builds

### Check build status:
- Green ✅: Build successful
- Red ❌: Build failed
- Yellow 🟡: Build in progress

### View build logs:
1. Click on the workflow run
2. Expand each step to see detailed logs
3. Look for error messages if build fails

## 🛠️ Customization

### Modify Python versions:
Edit `.github/workflows/build-windows.yml`:
```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']  # Add/remove versions
```

### Add more executables:
Add new PyInstaller spec creation in the workflow:
```yaml
- name: 📝 Create Additional Spec
  run: |
    # Create spec for new application
    $newSpec = @"
    # Your PyInstaller spec here
    "@
    $newSpec | Out-File -FilePath "build_new_app.spec"
```

### Change build triggers:
Modify the `on:` section:
```yaml
on:
  push:
    branches: [ main ]  # Only main branch
  # Remove other triggers if not needed
```

## 🚨 Troubleshooting

### Common issues:

1. **Build fails on dependencies**:
   - Check `requirements_windows.txt` exists
   - Ensure all packages are Windows-compatible

2. **PyInstaller errors**:
   - Review hidden imports in spec files
   - Check for missing data files

3. **Large executable sizes**:
   - Add exclusions to PyInstaller spec
   - Use UPX compression (already enabled)

4. **Permission errors**:
   - Check GitHub Actions permissions
   - Ensure repository settings allow Actions

### Getting help:
- Check the **Actions** logs for detailed error messages
- Look at the **Issues** tab for known problems
- Review PyInstaller documentation for spec file options

## 🎯 Best Practices

### Version tagging:
```bash
# Create version tags for releases
git tag v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Branch protection:
- Require pull request reviews
- Require status checks (builds) to pass

### Artifact retention:
- Build artifacts kept for 30 days
- Release distributions kept for 90 days

## 🎉 Success!

Once set up, you'll have:
- ✅ Automatic Windows executable builds
- ✅ Cross-platform compatibility testing
- ✅ Professional distribution packages
- ✅ Release management
- ✅ No need for Windows development machine

Your users will be able to download and run the executables directly on Windows without any Python installation! 🚀
