# How to Connect VS Code to Google Colab Pro

## One-Time Setup (5 minutes)

### 1. Install VS Code Extension

In VS Code, install:
- **Remote - SSH** (by Microsoft)

### 2. Get ngrok Account (Free)

1. Go to https://ngrok.com/
2. Sign up (free)
3. Go to https://dashboard.ngrok.com/get-started/your-authtoken
4. Copy your auth token

---

## Every Time You Start a Colab Session

### 1. Open Colab Notebook

1. Go to https://colab.research.google.com/
2. Upload `notebooks/colab_vscode_setup.ipynb`
3. Change runtime: **Runtime → Change runtime type → T4 GPU (or better)**

### 2. Run Setup Cells

**Cell 1:** Install dependencies
```python
!pip install colab-ssh --upgrade -q
```

**Cell 2:** Start SSH server

**Option A - Using ngrok (faster, recommended):**
```python
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")  # Paste your token
from colab_ssh import launch_ssh
launch_ssh(ngrok, password="colab")
```

**Option B - Using Cloudflare (slower, no token needed):**
```python
from colab_ssh import launch_ssh_cloudflared
launch_ssh_cloudflared(password="colab")
```

### 3. Copy SSH Command

The output will show something like:
```
ssh root@0.tcp.ngrok.io -p 12345
```

Copy this command!

### 4. Connect from VS Code

1. In VS Code: **Cmd+Shift+P** (Mac) or **Ctrl+Shift+P** (Windows)
2. Type: **Remote-SSH: Connect to Host**
3. Select **Add New SSH Host**
4. Paste the SSH command from Colab
5. When prompted for password, type: **colab**

### 5. You're Connected!

- Open Terminal in VS Code (it's running on Colab GPU!)
- Navigate to your project: `cd /content`
- Clone your repo or upload files
- Start coding in VS Code with Colab GPU power!

---

## Important Notes

 **Session Limits:**
- Colab Pro sessions last ~12 hours max
- Sessions timeout if idle too long
- You'll need to re-run the setup notebook each time

 **Tips:**
- Keep the last cell running in Colab to prevent timeout
- Save/commit your work frequently
- Use GitHub to sync code between sessions

 **Workflow:**
1. Start Colab session
2. Run setup notebook
3. Connect VS Code
4. Code in VS Code
5. Commit/push to GitHub before session ends

---

## Troubleshooting

**"Connection refused"**
- Make sure the Colab notebook is still running
- Re-run the SSH setup cell

**"Password doesn't work"**
- Password is: `colab` (lowercase)
- Make sure you set it correctly in launch_ssh()

**"Session timed out"**
- Colab killed your session
- Start a new Colab session and reconnect

**"Can't see my files"**
- Files are in `/content` on Colab
- Navigate there: `cd /content`
- Clone your repo or upload files

---

## Alternative: Simpler but Less Integrated

If the SSH setup is too annoying, you can also:

1. **Code locally in VS Code**
2. **Push to GitHub**
3. **Pull in Colab and run training**
4. **Download results**

Many people prefer this workflow because it's more stable.
