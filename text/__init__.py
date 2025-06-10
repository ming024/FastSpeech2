""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from string import punctuation
from text.symbols import symbols
from .en_phonological_features_new import phonological_features,default_feature
import numpy as np 
import torch
import torch.nn as nn

# Mappings from symbol to numeric ID and vice versa:
#_symbol_to_id = {s: i for i, s in enumerate(symbols)}
#_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

#for german 
def text_to_sequence(text, cleaner_names):
    processed_phonemes = []
    feature_list = []

    m = _curly_re.match(text)
    phones = m.group(2).strip().split()

    for p in phones:
        #if p in diphthongs_map:
            #for single_phoneme in diphthongs_map[p]:
                #processed_phonemes.append(single_phoneme)
                #feature = phonological_features.get(single_phoneme, default_feature)
                #feature_list.append(feature)
        #else:
        processed_phonemes.append(p)
        feature = phonological_features.get(p, default_feature)
        #print(f"Phoneme: {p}, Feature shape: {np.array(feature).shape}, Feature type: {type(feature)}")
        feature_list.append(feature)

    features = np.array(feature_list, dtype=np.float32)
    features = torch.from_numpy(features).float()       
    return features



def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"


