from functools import partial
from warnings import warn

from spacy.language import Language

from asent.component import Asent
from asent.lang.emoji import LEXICON as E_LEXICON
from asent.utils import LEXICON_PATH, components, lexicons, read_lexicon

lang_path = LEXICON_PATH / ".." / "lang"
langs = [lang.stem for lang in lang_path.glob("*.py") if len(lang.stem) < 4]


def create_xx_sentiment_component(
    nlp: Language,
    name: str,
    lang: str,
    force: bool,
) -> Asent:
    """Allows the sentiment pipe to be added to a spaCy pipe using
    nlp.add_pipe("asent_{language id}_v1"). Note that this is overwritten by
    languages which have a specified default, e.g. the case for en, da, sv.

    note that this function uses a lexicon that is automatically
    constructed. We therefore recommend examines the output and
    adjusting the lexicon. For more information on how the lexicon was
    constructed see; Chen, Y., & Skiena, S. (2014). Building Sentiment
    Lexicons for All Major Languages.

    Args:
        nlp: The spaCy language object to add the sentiment pipe to.
        name: The name of the sentiment pipe.
        lang: The language id of the sentiment pipe.
        force: Whether to force the sentiment pipe to be added to the language.

    Returns:
        Asent: The sentiment pipe.
    """

    msg = (
        f"'asent_{lang}_v1' uses a lexicon that is automatically constructed. We "
        + "therefore recommend examines the output and adjusting the lexicon. "
        + "For more information on how the lexicon was constructed see; Chen, Y., & "
        + "Skiena, S. (2014). Building Sentiment Lexicons for All Major Languages."
    )
    warn(msg)

    lex = lexicons.get(f"lexicon_{lang}_chen_skiena_2014_v1.txt")
    lex.update(E_LEXICON)

    return Asent(
        nlp,
        name=name,
        lexicon=lex,
        intensifiers={},
        negations={},
        contrastive_conjugations={},
        lowercase=True,
        lemmatize=False,
        force=force,
    )


_create_xx_component = {}

for lex_path in LEXICON_PATH.glob("*_lexicon_chen_skiena_2014_v1.txt"):
    lang = lex_path.stem.split("_")[0]

    # register the Chen Skiena (2014) lexicons
    lexicons.register(
        f"lexicon_{lang}_chen_skiena_2014_v1.txt",
        func=read_lexicon(LEXICON_PATH / f"{lang}_lexicon_chen_skiena_2014_v1.txt"),
    )

    # if there is no specified default for the language specify the default to be the
    # autogenerated lexicon.
    if lang in langs:
        continue
    lexicons.register(
        f"lexicon_{lang}_v1",
        func=lexicons.get(f"lexicon_{lang}_chen_skiena_2014_v1.txt"),
    )
    lex = lexicons.get(f"lexicon_{lang}_v1")
    lex.update(E_LEXICON)

    _create_xx_component[lang] = partial(create_xx_sentiment_component, lang=lang)

    Language.factory(
        f"asent_{lang}_v1",
        default_config={"force": True},
        func=_create_xx_component[lang],
    )

    components.register(f"asent_{lang}_v1", func=_create_xx_component[lang])
