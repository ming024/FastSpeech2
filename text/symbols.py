""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. 

For the three languages in my dissertation, Gitksan (git), Kanien'kéha (moh), & SENĆOŦEN (str), mappings were used from the g2p library. If the language is supported by g2p,
it's recommended to add it in the same fashion as below

"""

from text import cmudict, pinyin
from g2p.mappings.langs import MAPPINGS_AVAILABLE
from g2p.transducer import CompositeTransducer, Transducer
from g2p import make_g2p
from nltk.tokenize import RegexpTokenizer
from unicodedata import normalize
from epitran.flite import Flite

FLITE = Flite()
ARPA_IPA = [x for x in FLITE.arpa_map.values() if x]

_pad = "_"
_punctuation = "!(),.;? ':"
_moh_punctuation = "!(),.;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["sp", "spn", "sil", "@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ["@" + s for s in cmudict.valid_symbols]
_arpabet = [s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

MAPPINGS = {
    "git": {"norm": make_g2p("git", "git-equiv"), "ipa": make_g2p("git", "git-ipa")},
    "moh": {"norm": make_g2p("moh", "moh-equiv"), "ipa": make_g2p("moh", "moh-ipa")},
    "str": {"norm": make_g2p("str", "str-equiv"), "ipa": make_g2p("str", "str-ipa")},
}

IPA = {}

for lang in MAPPINGS.keys():
    if isinstance(MAPPINGS[lang]['ipa'], CompositeTransducer):
        chars = MAPPINGS[lang]["ipa"]._transducers[-1].mapping.mapping
    else:
        chars = MAPPINGS[lang]["ipa"].mapping.mapping
    IPA[lang] = [normalize('NFC', c['out']) for c in chars]

TOKENIZERS = {
    k: RegexpTokenizer("|".join(sorted(v, key=lambda x: len(x), reverse=True)))
    for k, v in IPA.items()
}

BASE_SYMBOLS = [_pad] + list(_special) + list(_punctuation) + _silences
MOH_BASE_SYMBOLS = [_pad] + list(_special) + list(_moh_punctuation) + _silences

# SYMBOLS = {k: v + BASE_SYMBOLS for k, v in IPA.items()} # moh uses different base
SYMBOLS = {}
SYMBOLS["moh"] = MOH_BASE_SYMBOLS + IPA["moh"]
SYMBOLS["git"] = BASE_SYMBOLS + IPA["git"]
SYMBOLS["str"] = BASE_SYMBOLS + IPA["str"]
SYMBOLS["eng"] = BASE_SYMBOLS + ARPA_IPA + _arpabet

# # Export all symbols:
CHARS = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    # + _moh_ipa
    # + _moh_orth
    + list(_letters)
    + _arpabet
    # + _pinyin
    + _silences
)
