# Canal Estoicismo - Generador Automático de Videos

Sistema automatizado para crear videos de YouTube sobre estoicismo usando diapositivas dinámicas con narración.

## Características

- **Generación narrativa**: Historias estoicas de 3-5 minutos divididas en escenas
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
│   ├── test_config.py
│   ├── test_narrative.py
│   ├── test_audio.py
│   ├── test_media.py
│   └── test_composer.py
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

### Google Colab (GPU T4 para generación de imágenes con Diffusers SDXL)

1. **Abrir notebook nuevo** en [Google Colab](https://colab.research.google.com/)
2. **Activar GPU T4**: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `GPU T4`
3. **Ejecutar celdas de instalación**:

```python
# Celda 1: Clonar e instalar dependencias
!git clone https://github.com/TU_USUARIO/canal-estoicismo.git
%cd canal-estoicismo
!pip install -r requirements.txt diffusers torch accelerate transformers --quiet
```

```python
# Celda 2: Generar video con imágenes IA (SDXL Turbo - rápido en T4)
!python integration_test.py --theme control_dichotomy --output ./output_colab
```

```python
# Celda 3: Descargar resultado
from google.colab import files
files.download("./output_colab/stoic_integration_control_dichotomy_*.mp4")
```

**Tiempo estimado en Colab T4**: ~2-3 minutos para video 3-5 min (SDXL Turbo 2 pasos por imagen).

## Uso

### Generar un video individual

```bash
# Video completo (3-5 min)
python main.py generate --theme control_dichotomy

# Tema específico con duración personalizada
python main.py generate --theme memento_mori --duration 4 --scenes 6

# Modo borrador (2s por escena para testing rápido)
python main.py generate --theme amor_fati --draft
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