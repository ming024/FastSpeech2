""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin, german_mfa,english_mfa

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
#_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöü"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_silences = ["@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@" + s for s in cmudict.valid_symbols]
#_pinyin = ["@" + s for s in pinyin.valid_symbols]
#_ipa_german = ["@" + s for s in german_mfa.valid_symbols]
_ipa_english = ["@" + s for s in english_mfa.valid_symbols]
# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    #+ _arpabet
    #+ _pinyin
    + _ipa_english
    + _silences
)
