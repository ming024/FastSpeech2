""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

#TO_DO maybe change the pad
_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]
ara = ['yy', "ii1'", 'S', 'Z', "I0'", 'ii0', 'p', 'i0', 'hh', 'uu0', 'aa', 'UU0', 'U1', 'g', 'ZZ', 'ii1', "u0'", '$$', "I1'", 'DD', 'r', 'i', 'ww', 'dist', 'h', '*', 'H', 'D', '^^', 'ss', 'd', 'j', 'SH', 'q', 'J', 'zz', 'n', 'AA', "uu1'", "AA'", 'EE', "U0'", 'G', 'jj', 'TH', 'f', 'z', 'pp', 'SS', '$', "UU0'", 'l', "u1'", 'b', "i1'", 'U0', "aa'", "a'", '<', 'rr', 'tt', '<<', 'i1', 'nn', 'sil', 'v', 'x', 'w', 'Ah', "uu0'", "II0'", 'xx', 'II0', 'I1', 'E', 'TT', 'a', 't', 'uu1', "i0'", 'u', 'qq', 'gg', 'u0', 'kk', '**', 'k', 'I0', 'A', 'T', "UU1'", '-', 'm', 'll', 'dd', 'u1', 'ff', 'mm', '^', 'bb', 'AH', "ii0'", "A'", 'y', 'HH', 's']
_ara = ["@" + s for s in ara]

# Export all symbols:
#TO_DO
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
    + _ara
)
