# Canal Estoicismo - Generador Automático de Videos

Sistema automatizado para crear videos de YouTube sobre estoicismo usando diapositivas dinámicas con narración.

## Características

- **Generación narrativa**: Historias estoicas de 5-8 minutos divididas en escenas
- **Text-to-Speech**: Voces profundas y pausadas (edge-tts / gTTS)
- **Composición de video**: Ken Burns effect, subtítulos estilizados, 1080p
- **Pipeline modular**: Código limpio, testeado, listo para CI/CD

## Estructura del Proyecto

```
canal estoicismo/
├── config/                 # Configuración centralizada
│   ├── __init__.py
│   └── settings.py         # Video, audio, fuentes, narrativas, paths
├── src/
│   ├── narrative/          # Generación de guiones
│   │   ├── __init__.py
│   │   ├── story_generator.py
│   │   └── templates.py    # 10 plantillas estoicas
│   ├── audio/              # Text-to-Speech
│   │   ├── __init__.py
│   │   └── tts_engine.py
│   ├── media/              # Gestión de imágenes
│   │   ├── __init__.py
│   │   └── image_manager.py
│   ├── video/              # Composición final
│   │   ├── __init__.py
│   │   └── composer.py
│   └── __init__.py
├── tests/                  # Suite de pruebas completa
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_colab_env.py
│   ├── test_profiles.py
│   ├── test_config.py
│   ├── test_narrative.py
│   ├── test_audio.py
│   ├── test_media.py
│   └── test_composer.py
├── notebooks/               # Notebooks para Colab (ver docs/COLAB_A100.md)
│   ├── 00_entorno_A100.ipynb
│   ├── 01_verificacion_A100.ipynb
│   ├── 02_lote_reels.ipynb
│   └── 03_entrega.ipynb
├── scripts/                 # Ayudas de Colab + gate de aceptación
│   ├── colab_env.py
│   └── colab_check.py
├── docs/
│   └── COLAB_A100.md        # Arquitectura A100 y runbook
├── tools/
│   └── make_notebooks.py    # Regenera los .ipynb (valida sintaxis y flags)
├── .github/workflows/
│   └── ci.yml              # CI/CD con lint, tests, integración
├── main.py                 # CLI entry point
└── requirements.txt
```

## Instalación

### Local (Linux/macOS/Windows con WSL)

```bash
# Dependencias del sistema (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg imagemagick

# Dependencias Python
pip install -r requirements.txt
```

### Google Colab con A100 (ruta recomendada)

Los notebooks en [`notebooks/`](notebooks/) son la vía oficial. Cada uno se abre directamente desde GitHub y se verifica a sí mismo antes de gastar unidades:

```
00_entorno_A100.ipynb        entorno + GPU + Drive + tests + doctor
01_verificacion_A100.ipynb   2 imágenes SDXL reales, voz y vídeo de prueba (~2 min de GPU)
02_lote_reels.ipynb          lote de reels → Drive, con manifest verificado
03_entrega.ipynb             verificación, ZIP y reanudación de reels cortados
```

Abre el primero así:

```
https://colab.research.google.com/github/jorgeecort-boop/canal-estoicismo/blob/master/notebooks/00_entorno_A100.ipynb
```

1. `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → **A100 GPU**.
2. Añade el secreto `GEMINI_API_KEY` (icono de la llave) para guiones con LLM.
3. Ejecuta las celdas en orden.

Detalles completos en **[`docs/COLAB_A100.md`](docs/COLAB_A100.md)**: arquitectura, perfiles, persistencia en Drive, modelo de coste y diagnóstico.

Bootstrap alternativo de una sola celda (sin Drive, para terminal/CI): `!python setup_colab.py`.

## Uso

### Generar un video individual

```bash
# Video completo (5-8 min)
python main.py generate --theme control_dichotomy

# Tema específico con duración personalizada
python main.py generate --theme memento_mori --duration 5 --scenes 6

# Modo borrador (2s por escena para testing rápido)
python main.py generate --theme amor_fati --draft

# Voz del canal (default): edge-tts es-MX-JorgeNeural, grave y pausada
python main.py generate --theme control_dichotomy --scenes 3

# Voz Piper offline en CPU (instala requirements-piper.txt primero)
python main.py generate --theme control_dichotomy --scenes 3 --tts piper-cpu
```

### Generar lote de videos

```bash
# 5 videos con temas aleatorios
python main.py batch --count 5 --output-dir ./mis_videos

# Temas específicos
python main.py batch --count 3 --themes control_dichotomy impermanence virtud_ethics
```

### Probar el pipeline

```bash
# Test completo en 2 segundos (draft mode)
python main.py test --theme control_dichotomy
```

## Temas Disponibles

| Tema | Filósofo | Enfoque |
|------|----------|---------|
| `control_dichotomy` | Epicteto | Dicotomía del control |
| `impermanence` | Marco Aurelio | Impermanencia |
| `virtue_ethics` | Séneca | Virtud como único bien |
| `amor_fati` | Nietzsche/Estoicos | Amar el destino |
| `memento_mori` | Estoicos | Recordar la muerte |
| `inner_fortress` | Marco Aurelio | Fortaleza interior |
| `adversity_growth` | Séneca | Adversidad como maestra |
| `present_moment` | Marco Aurelio/Epicteto | Poder del ahora |
| `ego_death` | Epicteto/Marco Aurelio | Muerte del ego |
| `cosmic_perspective` | Marco Aurelio | Vista desde arriba |

## Configuración

Edita `config/settings.py` o usa variables de entorno:

```python
from config import get_settings

settings = get_settings(
    debug=True,
    draft_mode=True,  # Videos de prueba rápidos
)
```

## Testing

```bash
# Tests unitarios
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov=config --cov-report=html

# Solo tests de narrativa
pytest tests/test_narrative.py -v
```

## CI/CD

El pipeline (`.github/workflows/ci.yml`) ejecuta en cada push:

1. **Lint & Type Check**: Ruff + MyPy
2. **Unit Tests**: pytest con cobertura
3. **Integration Test**: Render de prueba (2s draft mode)
4. **Build Package**: Solo en main branch

## Arquitectura de Datos

### Scene
```python
Scene(
    scene_number=1,
    text_overlay="Frase corta en pantalla",
    voiceover_text="Narración completa para TTS",
    image_prompt="Prompt para imagen de fondo",
    estimated_duration=12.5,
)
```

### Story
```python
Story(
    theme="control_dichotomy",
    title="La Dicotomía del Control",
    philosopher="Epicteto",
    scenes=[...],
    total_estimated_duration=180.0,
)
```

## Extensibilidad

- **Nuevos temas**: Añadir a `src/narrative/templates.py`
- **Nuevos proveedores TTS**: Extender `TTSEngine` en `src/audio/tts_engine.py`
- **Nuevas fuentes de imágenes**: Implementar en `ImageManager`
- **Efectos de video**: Modificar `_build_kenburns_filter` en `composer.py`

## Licencia

MIT - Libre para uso personal y comercial.
