"""Stoic story templates and narrative structures."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StoicTemplate:
    """Template for a stoic narrative structure."""
    theme: str
    title: str
    philosopher: str
    hook: str
    core_teaching: str
    practical_application: str
    closing_reflection: str
    image_keywords: tuple[str, ...]


STOIC_TEMPLATES: dict[str, StoicTemplate] = {
    "control_dichotomy": StoicTemplate(
        theme="control_dichotomy",
        title="La Dicotomia del Control",
        philosopher="Epicteto",
        hook=(
            "Hay cosas que dependen de nosotros, y cosas que no dependen de nosotros. "
            "La felicidad comienza cuando dejamos de luchar contra lo segundo."
        ),
        core_teaching=(
            "Epicteto nos enseña en el Enquiridion que nuestras opiniones, impulsos, "
            "deseos y aversiones son nuestros. Pero nuestro cuerpo, propiedades, "
            "reputacion y cargos no lo son. El sabio invierte su energia solo en lo primero."
        ),
        practical_application=(
            "Hoy, ante cada preocupacion, preguntate: '¿Depende de mi?' "
            "Si la respuesta es no, suelta el apego al resultado. "
            "Si es si, actua con virtud y acepta lo que venga."
        ),
        closing_reflection=(
            "La libertad verdadera no esta en controlar el mundo, "
            "sino en gobernar tu propia mente. Ahi reside tu imperio invencible."
        ),
        image_keywords=(
            "marble statue epictetus", "chains breaking", "inner light",
            "stoic philosopher meditation", "ancient scroll"
        ),
    ),

    "impermanence": StoicTemplate(
        theme="impermanence",
        title="La Impermanencia de Todas las Cosas",
        philosopher="Marco Aurelio",
        hook=(
            "Todo fluye. Nada permanece. Lo que hoy amas, manana sera polvo. "
            "¿Por que aferrarte a lo que ya se esta yendo?"
        ),
        core_teaching=(
            "En sus Meditaciones, Marco Aurelio escribe: 'Observa como todo cambia, "
            "y como la naturaleza ama transformar lo que es en lo que sera.' "
            "La resistencia al cambio es la raiz del sufrimiento."
        ),
        practical_application=(
            "Practica la 'vision desde arriba': imagina tu vida desde la perspectiva "
            "de las estrellas. Ve como las civilizaciones nacen y mueren, "
            "como tus preocupaciones son granos de arena en la eternidad."
        ),
        closing_reflection=(
            "No pierdas el presente lamentando el pasado o temiendo al futuro. "
            "El unico momento que posees es este, ahora. Vivel con virtud."
        ),
        image_keywords=(
            "ruins ancient rome", "hourglass sand flowing", "seasons changing",
            "marble columns crumbling", "cosmic perspective stars"
        ),
    ),

    "virtue_ethics": StoicTemplate(
        theme="virtue_ethics",
        title="La Virtud como Unico Bien",
        philosopher="Seneca",
        hook=(
            "La riqueza se pierde, la salud falla, la fama se desvanece. "
            "Solo la virtud -sabiduria, justicia, coraje, templanza- es verdaderamente tuya."
        ),
        core_teaching=(
            "Seneca ensena en sus Cartas a Lucilio que los bienes externos son 'preferidos' "
            "pero no 'buenos'. Solo el caracter virtuoso es un bien incondicional. "
            "Un esclavo virtuoso es mas libre que un tirano vicioso."
        ),
        practical_application=(
            "Antes de cada decision, pregunta: '¿Esto me hace mas sabio, mas justo, "
            "mas valiente, mas moderado?' Si la respuesta es no, no lo hagas. "
            "Tu caracter es tu destino."
        ),
        closing_reflection=(
            "Al final de tu vida, no se medira por lo que tuviste, "
            "sino por quien fuiste. Cultiva la excelencia del alma, "
            "pues es lo unico que la muerte no puede arrebatar."
        ),
        image_keywords=(
            "seneca statue", "scales justice", "torch wisdom",
            "lion courage", "temperance water wine"
        ),
    ),

    "amor_fati": StoicTemplate(
        theme="amor_fati",
        title="Amor Fati: Ama Tu Destino",
        philosopher="Nietzsche / Estoicos",
        hook=(
            "No solo soporta lo necesario. Amalo. Cada obstaculo es el camino. "
            "Cada herida, una leccion. Cada perdida, una clarificacion."
        ),
        core_teaching=(
            "Los estoicos llamaban a esto 'seguimiento de la naturaleza'. "
            "Marco Aurelio: 'Lo que impide la accion, promueve la accion. "
            "Lo que se interpone en el camino, se vuelve el camino.' "
            "El obstaculo es la via."
        ),
        practical_application=(
            "Cuando algo 'malo' ocurra, di: 'Esto es para mi bien. "
            "¿Que virtud puedo ejercer aqui? ¿Que aprendo?' "
            "Transforma la adversidad en combustible para tu caracter."
        ),
        closing_reflection=(
            "El destino no te sucede; te revela. Amalo todo -lo dulce y lo amargo- "
            "porque todo es necesario para que seas quien estas llamado a ser."
        ),
        image_keywords=(
            "phoenix rising ashes", "storm calm center", "forge fire metal",
            "mountain climber summit", "lotus mud bloom"
        ),
    ),

    "memento_mori": StoicTemplate(
        theme="memento_mori",
        title="Memento Mori: Recuerda que Moriras",
        philosopher="Estoicos / Tradicion Medieval",
        hook=(
            "Podrias no despertar manana. ¿Vivirias hoy de la misma manera? "
            "La muerte no es el final; es el consejero que da urgencia a la virtud."
        ),
        core_teaching=(
            "Seneca: 'Ensayemos la muerte. Nada hay de malo en ella, "
            "pues nos libera de todos los males.' "
            "Epicteto: 'Ten siempre presente la muerte y el destierro, "
            "y todo lo que parece terrible, y nunca tendras un pensamiento servil.' "
            "La mortalidad clarifica lo esencial."
        ),
        practical_application=(
            "Cada noche, repasa tu dia como si fuera el ultimo. "
            "¿Fuiste justo? ¿Valiente? ¿Sabio? ¿Moderado? "
            "Corrige el rumbo mientras tienes tiempo. El tiempo es tu unico capital."
        ),
        closing_reflection=(
            "No temas a la muerte. Teme a una vida sin virtud. "
            "Quien vive cada dia como si fuera el ultimo, "
            "nunca tiene un dia incompleto."
        ),
        image_keywords=(
            "skull marble", "candle burning low", "ancient tomb",
            "clock stopped", "shadows lengthening"
        ),
    ),

    "inner_fortress": StoicTemplate(
        theme="inner_fortress",
        title="La Fortaleza Interior",
        philosopher="Marco Aurelio",
        hook=(
            "Nadie puede danarte sin tu consentimiento. "
            "Tu mente es una ciudadela que ninguna fuerza externa puede tomar."
        ),
        core_teaching=(
            "Marco Aurelio: 'La mente libre de pasiones es una fortaleza. "
            "El hombre no tiene refugio mas seguro.' "
            "Las impresiones externas golpean las murallas, "
            "pero el gobernante interior decide si abrir las puertas."
        ),
        practical_application=(
            "Cuando sientas ira, miedo, ansiedad: haz una pausa. "
            "Observa la impresion: 'Esto es solo una apariencia, no la realidad.' "
            "Elige tu respuesta. Ahi esta tu poder. Ahi esta tu libertad."
        ),
        closing_reflection=(
            "El mundo puede quitarte todo: salud, riqueza, reputacion, "
            "incluso a tus seres queridos. Pero nunca podra quitarte "
            "tu capacidad de elegir quien eres frente a lo que ocurre."
        ),
        image_keywords=(
            "fortress walls", "citadel mountain", "shield light",
            "inner sanctuary", "unshakeable pillar"
        ),
    ),

    "adversity_growth": StoicTemplate(
        theme="adversity_growth",
        title="La Adversidad como Maestra",
        philosopher="Seneca",
        hook=(
            "Un mar tranquilo no hace marineros habiles. "
            "El oro se prueba en el fuego. El caracter, en la dificultad."
        ),
        core_teaching=(
            "Seneca en 'De la Providencia': '¿Por que le suceden cosas malas a los buenos? "
            "Para que puedan demostrar su virtud.' "
            "La adversidad no es castigo; es entrenamiento. "
            "Dios (o la Naturaleza) endurece a quienes ama."
        ),
        practical_application=(
            "Reencuadra cada dificultad: 'Esto es un gimnasio para mi alma. "
            "¿Que musculo virtuoso estoy ejercitando? Paciencia, coraje, "
            "resiliencia, compasion?' Agradece el entrenamiento."
        ),
        closing_reflection=(
            "Mira hacia atras: tus mayores crecimientos vinieron de tus mayores dolores. "
            "Confia en el proceso. La presion crea diamantes. "
            "Tu eres el diamante en formacion."
        ),
        image_keywords=(
            "diamond formation pressure", "blacksmith hammer anvil",
            "tree roots rock", "storm oak standing", "refiner's fire"
        ),
    ),

    "present_moment": StoicTemplate(
        theme="present_moment",
        title="El Poder del Ahora",
        philosopher="Marco Aurelio / Epicteto",
        hook=(
            "El pasado ya no existe. El futuro aun no existe. "
            "Solo el presente es real. Y en el presente, solo tu juicio es libre."
        ),
        core_teaching=(
            "Marco Aurelio: 'Confina el presente. "
            "No te angusties por el futuro; llegaras a el, si es necesario, "
            "con las mismas armas de la razon que usas ahora contra el presente.' "
            "Epicteto: 'No te preocupes por el futuro; ocupate del presente.'"
        ),
        practical_application=(
            "Cuando la mente divague al pasado o futuro, vuelve suavemente: "
            "'Aqui. Ahora. ¿Que requiere virtud en este instante?' "
            "La vida se vive solo en el presente. No la pierdas en la imaginacion."
        ),
        closing_reflection=(
            "Tu vida no es la suma de tus anos, "
            "sino la calidad de tus momentos presentes. "
            "Haz de cada ahora una ofrenda a la excelencia."
        ),
        image_keywords=(
            "single water drop", "meditation stone", "zen garden",
            "present moment awareness", "still lake reflection"
        ),
    ),

    "ego_death": StoicTemplate(
        theme="ego_death",
        title="La Muerte del Ego",
        philosopher="Epicteto / Marco Aurelio",
        hook=(
            "Tu ego quiere reconocimiento, control, superioridad. "
            "Tu alma quiere verdad, virtud, conexion. Solo una puede ganar."
        ),
        core_teaching=(
            "Epicteto: 'Si quieres mejorar, contentate con parecer ignorante "
            "en cosas externas.' "
            "Marco Aurelio: '¿Que es la fama? El aplauso de gente que no se estima a si misma.' "
            "El ego es ruido; la virtud es senal."
        ),
        practical_application=(
            "Haz el bien sin testigos. Aprende sin presumir. "
            "Admite errores sin defenderte. Perdona sin esperar reciprocidad. "
            "Cada vez que el ego clama, elige la humildad. Es fuerza, no debilidad."
        ),
        closing_reflection=(
            "Cuando el ego muere, nace el sabio. "
            "Libre de la necesidad de ser 'alguien', "
            "por fin puedes ser verdaderamente alguien: virtuoso."
        ),
        image_keywords=(
            "mask falling", "true face revealed", "ego dissolution",
            "humble servant", "empty throne"
        ),
    ),

    "cosmic_perspective": StoicTemplate(
        theme="cosmic_perspective",
        title="La Perspectiva Cosmica",
        philosopher="Marco Aurelio",
        hook=(
            "Sube a las estrellas. Mira tu vida desde alli. "
            "Tus guerras, tus amores, tus ambiciones... "
            "¿Que son ante la eternidad?"
        ),
        core_teaching=(
            "Marco Aurelio, Meditaciones 7.48: 'Mira desde arriba: "
            "rebanos, ejercitos, labranzas, bodas, divorcios, nacimientos, muertes... "
            "Todo pequeno, todo efimero.' "
            "La vista desde arriba disuelve la ansiedad y ordena las prioridades."
        ),
        practical_application=(
            "Cuando un problema te abruma, haz el ejercicio de la 'vista desde arriba': "
            "Visualiza la Tierra desde el espacio. Luego tu ciudad. Luego tu calle. "
            "Luego tu habitacion. Luego tu. ¿Que importancia real tiene esto en 1000 anos?"
        ),
        closing_reflection=(
            "Somos polvo de estrellas consciente por un instante. "
            "Usa ese instante para brillar con virtud. "
            "El cosmos no te juzga; tu te juzgas. Se digno de tu conciencia."
        ),
        image_keywords=(
            "earth from space", "cosmic perspective", "stars galaxies",
            "pale blue dot", "astronaut overview effect"
        ),
    ),
}


def get_template(theme: str) -> StoicTemplate:
    """Get a template by theme name."""
    if theme not in STOIC_TEMPLATES:
        raise ValueError(f"Unknown theme: {theme}. Available: {list(STOIC_TEMPLATES.keys())}")
    return STOIC_TEMPLATES[theme]


def list_themes() -> list[str]:
    """List all available themes."""
    return list(STOIC_TEMPLATES.keys())