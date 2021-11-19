""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import SYMBOLS, FLITE
from features import get_features
from text.cleaners import CLEANERS

# Mappings from symbol to numeric ID and vice versa:
_SYMBOLS_TO_ID = {k: {s: i for i, s in enumerate(v)} for k, v in SYMBOLS.items()}
_ID_TO_SYMBOL = {k: {i: s for i, s in enumerate(v)} for k, v in SYMBOLS.items()}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, use_spe_features, language, keep_padding=False):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(
                CLEANERS[language](text), language, keep_padding=keep_padding
            )
            break
        sequence += _symbols_to_sequence(
            CLEANERS[language](m.group(1)), language, keep_padding=keep_padding
        )
        if use_spe_features:
            if language != "eng":
                sequence += _ipa_to_feature_sequence(
                    m.group(2), language, keep_padding=keep_padding
                )
            else:
                sequence += _arpabet_to_feature_sequence(
                    m.group(2), language, keep_padding=keep_padding
                )
        elif language == "eng":
            sequence += _arpabet_to_sequence(
                m.group(2), language, keep_padding=keep_padding
            )
        else:
            sequence += _ipa_to_sequence(
                m.group(2), language, keep_padding=keep_padding
            )
        text = m.group(3)

    return sequence


def sequence_to_text(sequence, lang):
    """Converts a sequence of IDs back to a string
    AP: This appears to just be a debugging util - it's not used elsewhere
    """
    result = []
    for symbol_id in sequence:
        if symbol_id in _ID_TO_SYMBOL[lang]:
            s = _ID_TO_SYMBOL[lang][symbol_id]
            # # Enclose ARPAbet back in curly braces:
            # if len(s) > 1 and s[0] == "@":
            #     s = "{%s}" % s[1:]
            result.append(s)
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, lang, keep_padding=False):
    try:
        return [
            _SYMBOLS_TO_ID[lang][s]
            for s in symbols
            if _should_keep_symbol(s, lang, keep_padding=keep_padding)
        ]
    except:
        breakpoint()


def _ipa_to_sequence(text, lang, keep_padding=False):
    return _symbols_to_sequence(text.split(), lang, keep_padding=keep_padding)


def _arpabet_to_feature_sequence(text, lang, keep_padding=False):
    symbols = [
        "".join(filter(lambda x: not x.isdigit(), s)).lower()
        for s in text.split()
        if _should_keep_symbol(s, lang, keep_padding=keep_padding)
    ]
    # TODO: no dipths!
    return get_features(
        [FLITE.arpa_map[x] if x in FLITE.arpa_map else x for x in symbols]
    )


def _ipa_to_feature_sequence(text, lang, keep_padding=False):
    symbols = [
        s
        for s in text.split()
        if _should_keep_symbol(s, lang, keep_padding=keep_padding)
    ]
    return get_features(symbols)


def _arpabet_to_sequence(text, lang, keep_padding=False):
    return [
        _SYMBOLS_TO_ID[lang][s]
        for s in text.split()
        if _should_keep_symbol(s, lang, keep_padding=keep_padding)
    ]


def _should_keep_symbol(s, lang, keep_padding=False):
    if keep_padding:
        return s in _SYMBOLS_TO_ID[lang]
    else:
        return s in _SYMBOLS_TO_ID[lang] and s != "_" and s != "~"
