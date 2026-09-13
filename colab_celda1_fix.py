# @title 1️⃣ Clonar e instalar con versiones compatibles (fix)
!git clone https://github.com/jorgeecort-boop/canal-estoicismo.git
%cd canal-estoicismo

# Instalar versiones compatibles que existan en PyPI
!pip install --quiet \
    "transformers==4.44.0" \
    "diffusers==0.31.0" \
    "moviepy==2.0.0.dev2" \
    "google-generativeai==0.8.3" \
    -r requirements.txt \
    aiohttp

# Verificar imports (usa las versiones que Colab ya tenga si son compatibles)
import torch, diffusers, transformers, moviepy
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ Diffusers: {diffusers.__version__}")
print(f"✅ Transformers: {transformers.__version__}")
print(f"✅ MoviePy: {moviepy.__version__}")
print(f"✅ CUDA: {'Sí' if torch.cuda.is_available() else 'No'}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else "⚠️ Sin GPU")