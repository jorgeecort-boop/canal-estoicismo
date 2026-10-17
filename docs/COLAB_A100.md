# Pipeline Colab A100 — canal-estoicismo

Documento de referencia de la arquitectura definitiva: **solo Colab, GPU A100, sin ComfyUI, sin Runpod, sin túneles.**

---

## 1. Decisión y alcance

| decisión | por qué |
|---|---|
| **Solo Colab** | Nada depende de un túnel externo que se pueda caer. Ninguna pieza necesita una conexión abierta desde tu PC. |
| **A100 40 GB** | Cabe SDXL completo (base + refiner + hires) residente en VRAM, sin cuantizar. |
| **Sin ComfyUI** | Un servidor de UI no aporta nada aquí: el pipeline llama a `diffusers` directamente desde Python. Menos piezas que puedan caerse. |
| **Sin Runpod** | No aporta valor si Colab ya da A100 dentro de la suscripción que pagas. |
| **Imágenes estáticas + Ken Burns** | El motor visual y el Ken Burns ya dan movimiento cinematográfico sin generar vídeo por IA: coste en unidades mucho menor y cero riesgo de artefactos (flicker/morphing). |
| **Voz `edge-tts es-MX-JorgeNeural`** | Es la voz validada del canal. No se cambia. |
| **Persistencia en Google Drive** | Las sesiones mueren. Los reels y — sobre todo — la caché de imágenes viven en Drive. |

La arquitectura es **opt-in y reversible**: el perfil se elige con un flag (`--profile`), así que el mismo repositorio sigue funcionando en una T4 o en local sin cambiar código.

---

## 2. Arquitectura del pipeline

```
                    ┌──────────────────────────────────────────────┐
                    │  Notebook 00  · entorno + gates              │
                    │  GPU · Drive · repo · deps · tests · doctor  │
                    └───────────────────────┬──────────────────────┘
                                            │  (una vez por sesión)
        ┌───────────────────────────────────▼───────────────────────────────────┐
        │                          main.py  (CLI)                               │
        │   --profile auto|a100|t4   ·   --format reel|landscape                │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
   1. GUION            ┌───────────────────▼────────────────────┐
      (LLM o          │ src/narrative/story_generator.py        │
       plantillas)    │  gemini → nvidia → openrouter → … →      │
                      │  plantillas locales                      │
                      └───────────────────┬────────────────────┘
                                          │  story.json  ──► Drive (reutilizable)
   2. VOZ + IMÁGENES  ┌───────────────────▼────────────────────┐
      (en paralelo)   │ src/audio/tts_engine.py      (edge-tts) │
                      │   es-MX-JorgeNeural · -8% · -15Hz        │
                      │   + EQ + compresión → -14 LUFS          │
                      │                                          │
                      │ src/media/image_manager.py   (SDXL)     │
                      │   base (832x1216, 30 pasos)              │
                      │     ↓ denoising_end=0.8                  │
                      │   refiner (8 pasos)                      │
                      │     ↓                                    │
                      │   hires fix img2img (1080x1920)          │
                      │   caché SHA256 en Drive                  │
                      └───────────────────┬────────────────────┘
                                          │  imágenes liberadas de VRAM
   3. COMPOSICIÓN     ┌───────────────────▼────────────────────┐
                      │ src/video/composer.py                    │
                      │   Ken Burns · subtítulos temporizados ·   │
                      │   motor visual (partículas) · cine ·      │
                      │   xfade · libx264 (A100 no tiene NVENC)   │
                      └───────────────────┬────────────────────┘
                                          │
                       output/<lote>/stoic_<tema>_<fecha>.mp4
                       output/<lote>/stoic_<tema>_<fecha>.story.json
                       output/<lote>/manifest.json
```

---

## 3. El sistema de perfiles

`config/profiles.py` es el único sitio donde se define "reel" y "A100". El perfil se aplica **antes** de cualquier flag explícito, así que `--profile a100 --upscale` sigue respetando tu decisión.

| ajuste | T4 (por defecto) | A100 |
|---|---|---|
| modelo | `sdxl-turbo` | `stable-diffusion-xl-base-1.0` |
| refiner | — | `stable-diffusion-xl-refiner-1.0` |
| pasos | 2 | 30 (+8 del refiner) |
| guidance | 0.0 | 6.5 |
| resolución nativa | 832x1216 | 832x1216 |
| hires fix | no | sí (1080x1920, fuerza 0.32) |
| `cpu_offload` | sí | **no** (todo residente: mucho más rápido) |
| preset libx264 | `slow` | `medium` |
| hilos de encoder | 4 | 0 (todos los vCPU) |
| NVENC | opcional | **no existe en A100** |

`--profile auto` detecta la GPU sola, así que el mismo comando funciona en A100 y en T4.

### Por qué la cadena base → refiner → hires

SDXL está entrenado a ~1 MP. Pedirle 1080x1920 (2 MP) de una vez produce extremidades duplicadas y composiciones rotas — no es un problema de pasos, es del modelo. La secuencia correcta es:

1. **base** a 832x1216 (1.01 MP, su resolución nativa), 30 pasos, `denoising_end=0.8`;
2. **refiner** termina el 20% restante del schedule (8 pasos efectivos);
3. **hires fix**: se sube a 1080x1920 con Lanczos y se vuelve a pasar por img2img con fuerza 0.32 (~12 pasos efectivos), que sustituye la suavidad del Lanczos por detalle real.

Todo esto es **solo `diffusers`**. No se usa Real-ESRGAN ni `basicsr`, que es la dependencia que rompe las sesiones de Colab limpias.

---

## 4. Persistencia y ahorro de unidades

El notebook 00 enlaza (`symlink`) estas rutas del repo a tu Drive:

| ruta del repo | destino en Drive | por qué |
|---|---|---|
| `assets/images/` | `Mi unidad/canal-estoicismo/assets/images/` | **la caché de imágenes**: lo más caro del pipeline |
| `output/` | `…/output/` | reels, `story.json`, manifiestos |
| `logs/` | `…/logs/` | diagnóstico de sesiones pasadas |
| `temp/` | *(local, no se enlaza)* | espacio de trabajo: enlazarlo solo ralentizaría Drive |

La clave de la caché incluye el perfil (modelo, pasos, tamaño, hires). Eso significa dos cosas:

- cambiar de perfil **no** te sirve una imagen vieja de `sdxl-turbo`;
- repetir el mismo reel **no** vuelve a pagar las imágenes ya generadas.

Si un reel se corta a medias, se reanuda en `03_entrega.ipynb` con `--story-json`: se salta el LLM y reutiliza las imágenes cacheadas.

---

## 5. Los notebooks

| notebook | cuándo | coste | qué garantiza |
|---|---|---|---|
| `00_entorno_A100.ipynb` | una vez por sesión | ~0 | GPU correcta, Drive montado, deps instaladas, **212 tests en verde**, doctor en PASS |
| `01_verificacion_A100.ipynb` | antes de cada lote | ~2 min de GPU | 2 imágenes SDXL completas, voz a -14 LUFS, MP4 1080x1920; imprime el coste medido por imagen |
| `02_lote_reels.ipynb` | cuando quieres publicar | proporcional al lote | el lote completo, cada reel verificado y con su manifiesto |
| `03_entrega.ipynb` | al terminar (o en sesión nueva) | ~0 | verifica, empaqueta un ZIP, y reanuda reels cortados |

Abiertos desde GitHub directamente:

```
https://colab.research.google.com/github/jorgeecort-boop/canal-estoicismo/blob/master/notebooks/00_entorno_A100.ipynb
https://colab.research.google.com/github/jorgeecort-boop/canal-estoicismo/blob/master/notebooks/01_verificacion_A100.ipynb
https://colab.research.google.com/github/jorgeecort-boop/canal-estoicismo/blob/master/notebooks/02_lote_reels.ipynb
https://colab.research.google.com/github/jorgeecort-boop/canal-estoicismo/blob/master/notebooks/03_entrega.ipynb
```

Los notebooks **no** contienen lógica propia: llaman a `scripts/colab_env.py`. Si algo del montaje falla, se arregla en el repo con un `git pull`, sin editar celdas.

---

## 6. Runbook

### Sesión nueva

1. `Entorno de ejecución > Cambiar tipo de entorno de ejecución > A100 GPU`.
2. Secretos (icono de la llave): `GEMINI_API_KEY` (recomendado), y opcionalmente `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`.
3. Ejecutar **`00_entorno_A100.ipynb`** entero. Debe terminar con dos `[PASS]`.
4. Ejecutar **`01_verificacion_A100.ipynb`**. Si sale `[PASS]`, mira el coste medido por imagen.
5. Ejecutar **`02_lote_reels.ipynb`** con la configuración que quieras.
6. Al terminar, **`03_entrega.ipynb`** para el ZIP y la verificación.

### Si la sesión se cae a mitad del lote

Nada importante se pierde: reels, guiones y caché de imágenes ya están en Drive.

1. Reejecuta `00_entorno_A100.ipynb` (vale con las celdas 1-5).
2. Abre `03_entrega.ipynb` y pon `RESUME = True` en la última celda: termina los reels que quedaron a medias usando su `.story.json` y las imágenes ya cacheadas.

---

## 7. Modelo de coste

El notebook 01 mide de verdad y guarda el resultado en `output/_checks/acceptance.json`:

- `cold_image_seconds` — la primera imagen, incluyendo descarga y carga de modelos (se paga una vez por sesión).
- `warm_image_seconds` — **el coste real por escena**, que es lo que hay que multiplicar.

Presupuesto de un reel de 10 escenas:

```
imágenes   = warm_image_seconds × 10
voz        = ~4 s por escena × 10
composición ≈ 1x-2x el tiempo real del vídeo (libx264 en 12 vCPU)
```

Un lote de 5 reels de 6 minutos son ~50 imágenes: es exactamente ahí donde se van las unidades, y por eso la caché de Drive importa tanto. Repetir un lote ya generado cuesta casi solo la composición.

---

## 8. Qué es estable y qué puedes tocar

**Estable (no lo cambies sin motivo):**

- la voz `edge-tts es-MX-JorgeNeural` con su cadena de audio a -14 LUFS;
- la caché de imágenes en Drive;
- los gates: si un test o el doctor fallan, algo va mal de verdad.

**Ajustable desde los notebooks, sin tocar código:**

| variable | efecto |
|---|---|
| `COUNT` | número de reels del lote |
| `DURATION` | minutos por reel |
| `SCENES` | **una imagen SDXL por escena: es el dial del coste** |
| `THEMES` | qué temas, en qué orden |
| `USE_LLM` | guion con LLM o con plantillas locales |
| `QUALITY` | `--cinematic` + `--visual-engine` |

**Ajustable en el repo** (`config/profiles.py`): pasos, guidance, fuerza del hires, preset de libx264.

---

## 9. Diagnóstico

| síntoma | causa | arreglo |
|---|---|---|
| `no CUDA GPU visible` | runtime sin GPU | `Entorno de ejecución > A100 GPU`, reejecutar 00 |
| `no CUDA GPU` pero hay A100 | Colab pidió reiniciar tras el pip install | reiniciar el entorno y reejecutar desde la celda 1 |
| `hires=NO` en el gate G2 | falló la descarga del pipeline img2img | reejecutar: normalmente es red |
| CUDA out of memory | perfil A100 en una GPU pequeña | el doctor lo detecta antes: usa `--profile t4` |
| el lote se queda en `0/5` | ningún proveedor de guion disponible | añade `GEMINI_API_KEY` o quita `USE_LLM` |
| proxy/Drive lento | Drive con miles de archivos pequeños | el caché de imágenes ya está en Drive a propósito: no lo borres |
| `NameError: name 'theme' is not defined` | bug histórico del batch, ya corregido | actualiza el repo (`git pull`) |

---

## 10. Uso sin notebooks (terminal / CI)

Todo lo que hacen los notebooks se puede ejecutar a mano:

```bash
# Entorno y permisos
python main.py doctor --format reel --profile auto --image-provider sdxl_local

# Aceptación completa (2 imágenes, voz y un vídeo de 2 s)
python scripts/colab_check.py --format reel

# Un reel
python main.py generate --theme amor_fati --scenes 10 --duration 6 \
  --output ./reel.mp4 --format reel --profile auto \
  --image-provider sdxl_local --cinematic --visual-engine --llm auto

# Un lote de 5
python main.py batch --count 5 --output-dir ./lote \
  --themes amor_fati memento_mori control_dichotomy inner_fortress present_moment \
  --duration 6 --scenes 10 --format reel --profile auto \
  --image-provider sdxl_local --cinematic --visual-engine --llm auto

# Suite de tests
python -m pytest -q
```

Regenerar los notebooks tras editar sus fuentes:

```bash
python tools/make_notebooks.py          # valida sintaxis y flags, y escribe los .ipynb
python tools/make_notebooks.py --check  # solo valida
```
