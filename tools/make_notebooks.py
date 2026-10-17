#!/usr/bin/env python3
"""make_notebooks.py — build notebooks/*.ipynb from the sources below.

Why a generator instead of hand-edited .ipynb files:
  * the notebook code is reviewed as Python, not as escaped JSON;
  * every code cell is compiled before it is written, so a syntax error can
    never reach Colab;
  * every `--flag` used in a `main.py` command is checked against the real
    argparse definitions, so a renamed flag breaks the build, not the run.

    python tools/make_notebooks.py            # write the notebooks
    python tools/make_notebooks.py --check     # compile/validate only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"
MAIN_PY = ROOT / "main.py"

COLAB_METADATA = {
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}


# ══════════════════════════════════════════════════════════════════════
# 00 — Entorno
# ══════════════════════════════════════════════════════════════════════

NB00 = [
    ("md", r"""# 00 — Entorno A100 (canal-estoicismo)

Este notebook **prepara la sesión y no genera vídeo**. Si algo está mal, se ve aquí y no después de gastar unidades.

## Antes de empezar

1. `Entorno de ejecución > Cambiar tipo de entorno de ejecución > A100 GPU` (requiere Pro/Pro+).
2. Abre el panel de secretos (icono de la llave) y crea las claves que tengas:
   - `GEMINI_API_KEY` — recomendado, guiones largos de 5-8 min (gratis en https://aistudio.google.com/apikey)
   - `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` — opcionales, respaldo
3. Ejecuta las celdas **en orden**. Si la sesión se cae, vuelve a ejecutar solo este notebook.

## Qué se guarda y qué no

| ruta | dónde vive | sobrevive a la sesión |
|---|---|---|
| `output/*.mp4` y `*.story.json` | tu Drive | sí |
| `assets/images/*.jpg` (caché SDXL) | tu Drive | sí |
| `logs/` | tu Drive | sí |
| `temp/`, paquetes pip | /content (local) | no |

La caché de imágenes es lo que más unidades ahorra: si un reel se corta a medias, al repetirlo las imágenes ya generadas **no se vuelven a pagar**.
"""),

    ("code", r"""#@title 1. Comprobar la GPU (debe decir A100) { display-mode: "form" }
import subprocess

smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print(smi.stdout.strip() or smi.stderr.strip())

try:
    import torch
except ImportError as exc:
    raise SystemExit(f"torch no está disponible en este entorno: {exc}")

if not torch.cuda.is_available():
    raise SystemExit(
        "No hay GPU CUDA visible.\n"
        "Ve a Entorno de ejecución > Cambiar tipo de entorno de ejecución > A100 GPU,\n"
        "espera al reinicio y vuelve a ejecutar desde la celda 1."
    )

GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
print(f"\nGPU   : {GPU_NAME}")
print(f"VRAM  : {VRAM_GB:.1f} GB")
print(f"torch : {torch.__version__}  |  CUDA {torch.version.cuda}")

if VRAM_GB < 30:
    print(f"\n[AVISO] {VRAM_GB:.0f} GB no es una A100 (40 GB).")
    print("        El perfil 'a100' (SDXL full + refiner + hires) necesita ~24 GB de VRAM.")
    print("        Si estás en una T4, usa --profile t4 en los notebooks siguientes.")
else:
    print("\n[OK] A100 lista.")
"""),

    ("code", r"""#@title 2. Montar Google Drive (persistencia) { display-mode: "form" }
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/canal-estoicismo")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print("Drive listo:", DRIVE_ROOT)
existing = sorted(p.name for p in DRIVE_ROOT.iterdir())
print("Contenido:", existing or "(vacío)")
"""),

    ("code", r"""#@title 3. Traer el repositorio { display-mode: "form" }
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/jorgeecort-boop/canal-estoicismo.git"
PROJECT_DIR = Path("/content/canal-estoicismo")

if (PROJECT_DIR / ".git").exists():
    code = subprocess.run(["git", "-C", str(PROJECT_DIR), "pull", "--ff-only"]).returncode
    print(f"\ngit pull -> {code}")
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(PROJECT_DIR)], check=True)

commit = subprocess.run(
    ["git", "-C", str(PROJECT_DIR), "log", "-1", "--oneline"],
    capture_output=True, text=True,
).stdout.strip()
print("commit:", commit)
"""),

    ("code", r"""#@title 4. Cargar los helpers del proyecto { display-mode: "form" }
import sys
from pathlib import Path

PROJECT_DIR = Path("/content/canal-estoicismo")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scripts.colab_env as env

assert env.PROJECT_DIR.exists(), "falta el repositorio: ejecuta la celda 3"
print("proyecto:", env.PROJECT_DIR)
print("drive   :", env.DRIVE_ROOT)
"""),

    ("code", r"""#@title 5. Enlazar output/ y la caché de imágenes a Drive { display-mode: "form" }
for line in env.link_persistence():
    print(" ", line)

# Comprobación real: escribir en el repo debe escribir en Drive.
probe = env.PROJECT_DIR / "output" / ".drive_check"
probe.parent.mkdir(parents=True, exist_ok=True)
probe.write_text("ok", encoding="utf-8")
assert (env.DRIVE_ROOT / "output" / ".drive_check").exists(), "el enlace a Drive no funciona"
probe.unlink()
print("\n[OK] output/, assets/images/ y logs/ escriben directamente en Drive.")
"""),

    ("code", r"""#@title 6. Instalar dependencias (el torch de Colab NO se toca) { display-mode: "form" }
env.banner("Paquetes del sistema")
env.apt_install(env.SYSTEM_DEPS)

env.banner("Paquetes de Python")
env.pip_install(env.PYTHON_DEPS)

print("\n[OK] Instalación terminada.")
print("Si Colab pide reiniciar el entorno de ejecución, hazlo y vuelve a ejecutar desde la celda 1.")
"""),

    ("code", r"""#@title 7. Verificar torch y las versiones clave { display-mode: "form" }
info = env.check_torch()
for key, value in info.items():
    print(f"  {key:<15} {value}")

env.gate(
    "CUDA disponible",
    bool(info.get("cuda_available")),
    "sin CUDA el perfil A100 no puede generar imágenes",
)

import importlib

for module in ("diffusers", "transformers", "accelerate", "moviepy", "edge_tts", "safetensors"):
    try:
        loaded = importlib.import_module(module)
        print(f"  {module:<13} {getattr(loaded, '__version__', 'ok')}")
    except Exception as exc:
        env.gate(f"import {module}", False, f"{type(exc).__name__}: {exc}")

print("\n[OK] Dependencias cargadas.")
"""),

    ("code", r"""#@title 8. Cargar los secretos (API keys) { display-mode: "form" }
found = env.load_secrets()

if not (found.get("GEMINI_API_KEY") or found.get("GOOGLE_API_KEY")):
    print("\n[AVISO] Sin GEMINI_API_KEY el lote usará plantillas locales:")
    print("        guiones más cortos y menos variados. El pipeline funciona igual.")

print("\n[OK] Secretos cargados (los valores nunca se imprimen).")
"""),

    ("code", r"""#@title 9. GATE — suite de tests completa { display-mode: "form" }
code = env.run([sys.executable, "-m", "pytest", "-q"], cwd=env.PROJECT_DIR)
env.gate(
    "suite de tests",
    code == 0,
    "no continúes hasta que la suite pase: es la garantía de que el código no está roto",
)
"""),

    ("code", r"""#@title 10. GATE — doctor (entorno + perfil efectivo) { display-mode: "form" }
code = env.run(
    [
        sys.executable, "main.py", "doctor",
        "--format", "reel",
        "--profile", "auto",
        "--image-provider", "sdxl_local",
    ],
    cwd=env.PROJECT_DIR,
)
env.gate("doctor", code == 0, "revisa las líneas [FAIL] del doctor de arriba")
"""),

    ("md", r"""## Entorno listo

Si las dos últimas celdas dicen `[PASS]`, la sesión está preparada.

Siguiente paso: **`01_verificacion_A100.ipynb`** (comprueba de verdad SDXL full, la voz y la composición) y luego **`02_lote_reels.ipynb`**.

### Cómo ahorrar unidades

- Ejecuta la verificación **antes** de cada lote largo: si algo va a fallar, que falle en 2 minutos y no en 40.
- No borres `assets/images/` en Drive: es la caché que evita volver a pagar imágenes.
- Si un reel se corta, reanúdalo con `03_entrega.ipynb` usando el `.story.json` guardado.
- El `--profile auto` detecta la GPU sola: en A100 usa SDXL full + hires; en T4 vuelve a sdxl-turbo.
"""),
]


# ══════════════════════════════════════════════════════════════════════
# 01 — Verificación
# ══════════════════════════════════════════════════════════════════════

NB01 = [
    ("md", r"""# 01 — Verificación real en la A100

Se ejecuta **después** del notebook 00 y **antes** de cualquier lote.

Genera 2 imágenes SDXL completas, una locución y un vídeo de prueba de 2 s: lo suficiente para saber que todo funciona, y barato en unidades.

| gate | qué comprueba |
|---|---|
| G0 | GPU, VRAM, torch, diffusers |
| G1 | perfil efectivo (SDXL full, 30 pasos, hires, threads) |
| G2 | imagen final == canvas 1080x1920, y el coste real por imagen |
| G3 | voz `es-MX-JorgeNeural` masterizada a -14 LUFS |
| G4 | MP4 1080x1920 con audio y subtítulos |

Si algo falla, la celda se pone en rojo y **no hay que lanzar el lote**.
"""),

    ("code", r"""#@title 0. Preflight de la sesión { display-mode: "form" }
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path("/content/canal-estoicismo")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scripts.colab_env as env

if not (env.PROJECT_DIR / ".git").exists() or not (env.PROJECT_DIR / "output").exists():
    raise SystemExit("Sesión nueva o incompleta: ejecuta primero 00_entorno_A100.ipynb.")

commit = subprocess.run(
    ["git", "-C", str(env.PROJECT_DIR), "log", "-1", "--oneline"],
    capture_output=True, text=True,
).stdout.strip()
print("proyecto:", env.PROJECT_DIR)
print("commit  :", commit)
"""),

    ("code", r"""#@title 1. GATE — aceptación end-to-end { display-mode: "form" }
code = env.run(
    [sys.executable, "scripts/colab_check.py", "--format", "reel"],
    cwd=env.PROJECT_DIR,
)
env.gate(
    "aceptación A100",
    code == 0,
    "corrige los [FAIL] de arriba antes de gastar unidades en un lote",
)
"""),

    ("code", r"""#@title 2. Coste medido y proyección del lote { display-mode: "form" }
import json

report_path = env.PROJECT_DIR / "output" / "_checks" / "acceptance.json"
if not report_path.exists():
    raise SystemExit("No hay informe de aceptación: la celda 1 no llegó a terminar bien.")

report = json.loads(report_path.read_text(encoding="utf-8"))
gates = {g["name"]: g["seconds"] for g in report.get("gates", [])}

warm = report.get("warm_image_seconds") or 0.0
cold = report.get("cold_image_seconds") or 0.0
tts_seconds = gates.get("Voice") or 0.0

canvas = report["canvas"]
print(f"GPU                : {report['device']} ({report['vram_gb']} GB)")
print(f"canvas             : {canvas[0]}x{canvas[1]}")
print(f"imagen en frío     : {cold:.1f} s  (incluye cargar los modelos: se paga una vez)")
print(f"imagen en caliente : {warm:.1f} s  <- el coste real por escena")
print(f"voz (una frase)    : {tts_seconds:.1f} s")

SCENES = 10
print(f"\nProyección para un reel de {SCENES} escenas:")
print(f"  imágenes    : ~{warm * SCENES / 60:.1f} min")
print(f"  voz         : ~{tts_seconds * SCENES / 60:.1f} min")
print("  composición : ~6-12 min (libx264 en 12 vCPU, aproximadamente 1x-2x tiempo real)")
print(f"  TOTAL       : ~{(warm * SCENES + 60 * 9) / 60:.0f} min por reel")

print("\nLas imágenes se cachean en Drive: repetir un reel no las vuelve a pagar.")
"""),

    ("md", r"""## Decisión

- **Todo en `[PASS]`** → adelante con `02_lote_reels.ipynb`.
- **Algún `[FAIL]`** → no lances el lote. Corrige eso primero.

### Qué significa cada fallo típico

| fallo | causa más probable | arreglo |
|---|---|---|
| G0 sin CUDA | runtime sin GPU o reinicio pendiente | Entorno de ejecución > A100 GPU, y reejecutar 00 |
| G2 tamaño ≠ canvas | perfil no aplicado | revisa que el doctor diga `--profile a100` |
| G2 `hires=NO` | falló la carga del pipeline img2img | vuelve a ejecutar: suele ser un fallo de red al descargar |
| G3 LUFS fuera de rango | cadena de audio alterada | revisa `src/audio/tts_engine.py` |
| G4 sin audio | ffmpeg sin códec aac | reejecuta la instalación de la celda 6 del notebook 00 |
"""),
]


# ══════════════════════════════════════════════════════════════════════
# 02 — Lote
# ══════════════════════════════════════════════════════════════════════

NB02 = [
    ("md", r"""# 02 — Lote de reels (A100)

Genera el lote completo. **Cada vídeo queda en Drive** junto a su `.story.json` y un `manifest.json`.

El notebook comprueba el doctor antes de empezar y avisa si falta el informe de verificación del notebook 01.

Ajusta la celda 1 para decidir cuántos reels quieres, la duración y las escenas: esas tres cifras son las que mandan en el consumo de unidades.
"""),

    ("code", r"""#@title 1. Configuración del lote { display-mode: "form" }
#@markdown Número de reels a generar:
COUNT = 5 #@param {type:"integer"}
#@markdown Duración objetivo por reel (minutos):
DURATION = 6.0 #@param {type:"number"}
#@markdown Escenas por reel (una imagen SDXL cada una):
SCENES = 10 #@param {type:"integer"}
#@markdown Temas, uno por vídeo y en orden:
THEMES = "amor_fati,memento_mori,control_dichotomy,inner_fortress,present_moment" #@param {type:"string"}
#@markdown Usar un LLM para el guion si hay API key:
USE_LLM = True #@param {type:"boolean"}
#@markdown Efectos cinematográficos y motor visual (partículas):
QUALITY = True #@param {type:"boolean"}

import time

THEME_LIST = [theme.strip() for theme in THEMES.split(",") if theme.strip()]
RUN_NAME = "lote_" + time.strftime("%Y%m%d_%H%M")

print("lote     :", RUN_NAME)
print("videos   :", COUNT)
print("duracion :", DURATION, "min |", SCENES, "escenas")
print("temas    :", THEME_LIST)
"""),

    ("code", r"""#@title 2. Preflight — doctor + última verificación { display-mode: "form" }
import json
import sys
from pathlib import Path

PROJECT_DIR = Path("/content/canal-estoicismo")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scripts.colab_env as env

if not (env.PROJECT_DIR / ".git").exists():
    raise SystemExit("Sesión nueva: ejecuta primero 00_entorno_A100.ipynb.")
if not (env.PROJECT_DIR / "output").is_symlink():
    raise SystemExit("output/ no apunta a Drive: vuelve a ejecutar 00_entorno_A100.ipynb.")

code = env.run(
    [
        sys.executable, "main.py", "doctor",
        "--format", "reel",
        "--profile", "auto",
        "--image-provider", "sdxl_local",
    ],
    cwd=env.PROJECT_DIR,
)
env.gate("doctor", code == 0, "no lances el lote sin el doctor en PASS")

report_path = env.PROJECT_DIR / "output" / "_checks" / "acceptance.json"
if report_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verdict = "OK" if report.get("passed") else "FALLÓ"
    print(f"\nÚltima verificación: {report.get('created_at')} | {report.get('device')} | {verdict}")
else:
    print("\n[AVISO] No hay informe de verificación.")
    print("        Ejecuta 01_verificacion_A100.ipynb antes de un lote largo.")
"""),

    ("code", r"""#@title 3. Ejecutar el lote { display-mode: "form" }
output_dir = env.DRIVE_ROOT / "output" / RUN_NAME
output_dir.mkdir(parents=True, exist_ok=True)
print("destino:", output_dir)

command = [
    sys.executable, "main.py", "batch",
    "--count", str(COUNT),
    "--output-dir", str(output_dir),
    "--themes", *THEME_LIST,
    "--duration", str(DURATION),
    "--scenes", str(SCENES),
    "--format", "reel",
    "--profile", "auto",
    "--image-provider", "sdxl_local",
]
if USE_LLM:
    command += ["--llm", "auto"]
if QUALITY:
    command += ["--cinematic", "--visual-engine"]

code = env.run(command, cwd=env.PROJECT_DIR)
env.gate("lote completado", code == 0, "mira los [FAIL] por vídeo en el log de arriba")
"""),

    ("code", r"""#@title 4. Verificar los vídeos y guardar el manifest { display-mode: "form" }
rows = env.verify_videos(output_dir, "reel")
env.gate("se generó al menos un vídeo", len(rows) > 0, "el lote no produjo ningún .mp4")

manifest = env.write_manifest(
    output_dir,
    rows,
    extra={
        "run": RUN_NAME,
        "count_requested": COUNT,
        "duration_min": DURATION,
        "scenes": SCENES,
        "themes": THEME_LIST,
        "with_llm": USE_LLM,
    },
)

good = [row for row in rows if row["has_audio"] and row["duration"] > 0]
print(f"\n[OK] {len(good)}/{len(rows)} reels verificados.")
for row in rows:
    print(f"  {row['name']}  {row['duration']:.1f}s  {row['size_mb']:.1f} MB")
"""),

    ("md", r"""## Siguiente paso

Abre **`03_entrega.ipynb`** para verificar el lote desde Drive, descargarlo en un ZIP y reanudar cualquier reel que se haya cortado.

### Si la sesión se cae a mitad

No pasa nada grave: las imágenes ya generadas están cacheadas en Drive.

1. Vuelve a ejecutar `00_entorno_A100.ipynb` (celdas 1-5 bastan: GPU, Drive, repo, helpers, enlaces).
2. Vuelve a lanzar este lote con **el mismo** `RUN_NAME` o continúa con `03_entrega.ipynb`.
3. Los guiones `.story.json` ya guardados permiten reanudar sin volver a llamar al LLM.
"""),
]


# ══════════════════════════════════════════════════════════════════════
# 03 — Entrega
# ══════════════════════════════════════════════════════════════════════

NB03 = [
    ("md", r"""# 03 — Entrega y reanudación

Trabaja sobre lo que ya hay en Drive: verifica el último lote, empaqueta un ZIP para descargar y **reanuda los reels cortados** reutilizando el guion y las imágenes cacheadas.

Se puede ejecutar en una sesión nueva sin volver a generar nada.
"""),

    ("code", r"""#@title 1. Localizar el último lote en Drive { display-mode: "form" }
import sys
from pathlib import Path

PROJECT_DIR = Path("/content/canal-estoicismo")
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scripts.colab_env as env

candidates = sorted(
    (path for path in (env.DRIVE_ROOT / "output").glob("lote_*") if path.is_dir()),
    key=lambda path: path.name,
)
if not candidates:
    raise SystemExit("No hay lotes en Drive todavía: ejecuta primero 02_lote_reels.ipynb.")

RUN_DIR = candidates[-1]
print("lotes encontrados:", [path.name for path in candidates])
print("lote a procesar  :", RUN_DIR)
"""),

    ("code", r"""#@title 2. Verificar los vídeos del lote { display-mode: "form" }
rows = env.verify_videos(RUN_DIR, "reel")
manifest = env.write_manifest(RUN_DIR, rows, extra={"verified": True})
print(f"\nmanifest: {manifest}")
"""),

    ("code", r"""#@title 3. Empaquetar un ZIP { display-mode: "form" }
import zipfile

zip_path = RUN_DIR / (RUN_DIR.name + ".zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for item in sorted(RUN_DIR.iterdir()):
        if item.suffix in (".mp4", ".json") and item.is_file():
            archive.write(item, item.name)

print(f"ZIP: {zip_path.name}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
for name in sorted(zipfile.ZipFile(zip_path).namelist()):
    print("  ", name)
"""),

    ("code", r"""#@title 4. Descargar el ZIP { display-mode: "form" }
from google.colab import files

files.download(str(zip_path))
print("Descarga iniciada:", zip_path.name)
print("Si el navegador la bloquea, el ZIP también está en tu Drive.")
"""),

    ("code", r"""#@title 5. Reanudar reels cortados (usa guion + caché) { display-mode: "form" }
#@markdown Ejecutar la reanudación ahora:
RESUME = False #@param {type:"boolean"}

orphans = []
for story in sorted(RUN_DIR.glob("*.story.json")):
    video = RUN_DIR / story.name.replace(".story.json", ".mp4")
    if not video.exists():
        orphans.append((story, video))

if not orphans:
    print("No hay reels cortados en este lote.")
else:
    print(f"{len(orphans)} reel(s) sin terminar:\n")
    for story, video in orphans:
        print(f"  guion: {story.name}")
        print(f"  salida: {video.name}")

    if RESUME:
        for story, video in orphans:
            print(f"\n--- reanudando {video.name} ---")
            code = env.run(
                [
                    sys.executable, "main.py", "generate",
                    "--story-json", str(story),
                    "--output", str(video),
                    "--format", "reel",
                    "--profile", "auto",
                    "--image-provider", "sdxl_local",
                    "--cinematic",
                    "--visual-engine",
                ],
                cwd=env.PROJECT_DIR,
            )
            print(f"  exit={code} -> {'OK' if code == 0 else 'revisar'}")
        rows = env.verify_videos(RUN_DIR, "reel")
        env.write_manifest(RUN_DIR, rows, extra={"resumed": True})
    else:
        print("\nMarca RESUME=True y vuelve a ejecutar esta celda para completarlos.")
        print("Reutiliza el guion (--story-json) y las imágenes ya cacheadas: es lo más barato.")
"""),

    ("md", r"""## Hecho

- Los reels están en `Mi unidad/canal-estoicismo/output/<lote>/`.
- Cada vídeo tiene su `.story.json`: reutilízalo para rehacerlo sin gastar LLM ni generar imágenes nuevas.
- `manifest.json` guarda la verificación de este lote (resolución, duración, audio).

### Antes de publicar

1. Mira cada reel completo (sobre todo el audio y los subtítulos).
2. Comprueba que la duración encaja con el formato que publicas.
3. Recuerda: el `.story.json` es el activo reutilizable. Guárdalo siempre.
"""),
]


NOTEBOOKS = {
    "00_entorno_A100.ipynb": NB00,
    "01_verificacion_A100.ipynb": NB01,
    "02_lote_reels.ipynb": NB02,
    "03_entrega.ipynb": NB03,
}


# ── validation ───────────────────────────────────────────────────────

def declared_flags() -> set[str]:
    """Every --flag argparse declares in main.py."""
    source = MAIN_PY.read_text(encoding="utf-8")
    return set(re.findall(r'"(--[a-z][a-z0-9-]+)"', source))


def validate(cells: list[tuple[str, str]], name: str) -> list[str]:
    """Compile every code cell and check main.py flags. Returns problems."""
    problems: list[str] = []
    allowed = declared_flags()
    for index, (kind, text) in enumerate(cells):
        if kind != "code":
            continue
        try:
            compile(text, f"{name}:cell{index}", "exec")
        except SyntaxError as exc:
            problems.append(f"{name} cell {index}: SyntaxError line {exc.lineno}: {exc.msg}")
        # Only police flags on lines that actually invoke main.py.
        for line in text.splitlines():
            if "main.py" not in line:
                continue
            for flag in re.findall(r"--[a-z][a-z0-9-]+", line):
                if flag not in allowed:
                    problems.append(f"{name} cell {index}: unknown main.py flag {flag}")
    return problems


def to_notebook(cells: list[tuple[str, str]]) -> dict:
    converted = []
    for kind, text in cells:
        lines = text.splitlines(keepends=True)
        if kind == "code":
            converted.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            })
        else:
            converted.append({"cell_type": "markdown", "metadata": {}, "source": lines})
    return {
        "cells": converted,
        "metadata": COLAB_METADATA,
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Colab notebooks")
    parser.add_argument("--check", action="store_true", help="Validate only, do not write")
    args = parser.parse_args()

    problems: list[str] = []
    for name, cells in NOTEBOOKS.items():
        problems += validate(cells, name)

    if problems:
        print("VALIDATION FAILED:")
        for problem in problems:
            print("  -", problem)
        return 1

    total_cells = sum(len(cells) for cells in NOTEBOOKS.values())
    print(f"Validated {len(NOTEBOOKS)} notebooks / {total_cells} cells — all code cells compile,")
    print(f"and every main.py flag exists in the parser ({len(declared_flags())} flags declared).")

    if args.check:
        return 0

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        target = NOTEBOOK_DIR / name
        notebook = to_notebook(cells)
        # Round-trip through JSON before writing: a notebook that does not
        # parse would make Colab fail with a confusing error.
        payload = json.dumps(notebook, indent=1, ensure_ascii=False)
        json.loads(payload)
        target.write_text(payload + "\n", encoding="utf-8")
        print(f"  wrote {target.relative_to(ROOT)}  ({len(cells)} cells)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
