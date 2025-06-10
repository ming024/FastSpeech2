import numpy as np 
import os 
from text.german_mfa import valid_symbols
from text.IPA_to_phonefeats_mapping import ipa_to_phonemefeats 

def extract_feature_categories(ipa_to_phonemefeats):
    """
    get feature_categories from ipa_to_phonemefeats
    """
    feature_categories = {}
    
    for features in ipa_to_phonemefeats.values():
        for key, value in features.items():
            if key not in feature_categories:
                feature_categories[key] = set()
            feature_categories[key].add(value)
    
    for key in feature_categories:
        feature_categories[key] = sorted(feature_categories[key])
    
    return feature_categories

def build_feature_to_index(feature_categories):
    """
    map index according to feature_categories
    """
    feature_to_index = {}
    index = 0
    for category, values in feature_categories.items():
        for value in values:
            feature_to_index[(category, value)] = index
            index += 1
    return feature_to_index

def find_feature(symbol):
    """
    map feature given mfa symbols
    """
    if symbol in ipa_to_phonemefeats.keys():
        feature = ipa_to_phonemefeats[symbol]
        #print (f"Feature of {symbol} is {feature}")
        return feature
    elif 'ː' in symbol:
        feature = ipa_to_phonemefeats[symbol.replace("ː","")]
        #print (f"Feature of {symbol} is {feature}")
        return feature
    elif symbol == "tʃ":
        feature = ipa_to_phonemefeats['t͡ʃ']
        #print (f"Feature of {symbol} is {feature}")
        return feature
    elif len(symbol) == 2:
        feature0 = ipa_to_phonemefeats[symbol[0]]
        feature1 = ipa_to_phonemefeats[symbol[1]]
        #print (f"Feature of {symbol} is split to \n {symbol[0]} {feature0} \n {symbol[1]} {feature1}")
        if feature0 is None or feature1 is None:
            print(f"Unknown symbols in combination: {symbol}")
            return None
        return feature0, feature1 
    else:
        print(f'unkown symbol{symbol}')
        return None 

def phoneme_features_to_onehot(features, vector_length, feature_to_index):
    """
    map feature to onehot vector
    """    
    one_hot_vector = np.zeros(vector_length, dtype=int)
    for key, value in features.items():
        if (key, value) in feature_to_index:
            one_hot_vector[feature_to_index[(key, value)]] = 1
    return one_hot_vector
        
def process_symbol(symbol, feature_to_index, vector_length):
    """
    one-hot vector for single or diphtongs
    """
    feature = find_feature(symbol)
    if isinstance(feature, tuple):
        vector0 = phoneme_features_to_onehot(feature[0], vector_length, feature_to_index)
        vector1 = phoneme_features_to_onehot(feature[1], vector_length, feature_to_index)
        return (vector0, vector1)
    elif isinstance(feature, dict):
        return phoneme_features_to_onehot(feature, vector_length, feature_to_index)
    else:
        print(f"Cannot convert unknown symbol to one-hot vectors: {symbol}")
        return None

def main():
    # get categories
    feature_categories = extract_feature_categories(ipa_to_phonemefeats)
    
    # index
    feature_to_index = build_feature_to_index(feature_categories)
    vector_length = len(feature_to_index)
    print(f"Vector length: {vector_length}")

    # get vectors 
    for symbol in valid_symbols:
        result = process_symbol(symbol, feature_to_index, vector_length)
        if result is not None:
            if isinstance(result, tuple):
                print(f"One-hot vectors for {symbol}:")
                print(f"First part:\n{result[0]}\nSecond part:\n{result[1]}")
            else:
                print(f"One-hot vector for {symbol}:\n{result}")

# write to a file 
def write_features_to_file(filename, valid_symbols, feature_to_index, vector_length):
    # 获取当前脚本所在目录
    output_path = os.path.join(os.getcwd(), filename)
    processed_symbols = set()  # 用于跟踪已处理的符号
    
    with open(output_path, 'w') as f:
        f.write('phonological_features = {\n')
        for symbol in valid_symbols:
            if symbol in processed_symbols:  # 跳过已处理的符号
                continue
            
            result = process_symbol(symbol, feature_to_index, vector_length)
            if result is not None:
                if isinstance(result, tuple):  # Handling diphthongs or combined symbols
                    for idx, char in enumerate(symbol):
                        if char not in processed_symbols:  # 确保组合中的单个字符未被处理过
                            vector_str = ', '.join(map(str, result[idx].tolist()))
                            # Write each part of the combined symbol on a new line
                            f.write(f'    "{char}": [{vector_str}],  # {len(result[idx])}维\n')
                            processed_symbols.add(char)  # 标记为已处理
                else:  # Handling single phonemes
                    vector_str = ', '.join(map(str, result.tolist()))
                    f.write(f'    "{symbol}": [{vector_str}],  # {len(result)}维\n')
                    processed_symbols.add(symbol)  # 标记为已处理
        f.write('}\n')



if __name__ == "__main__":
    
    feature_categories = extract_feature_categories(ipa_to_phonemefeats)
    
    # index
    feature_to_index = build_feature_to_index(feature_categories)
    vector_length = len(feature_to_index)
    print(f"Vector length: {vector_length}")

    # Call main process
    main()
    
    # Write features to file in the current directory
    write_features_to_file('phonological_features_2.txt', valid_symbols, feature_to_index, vector_length)

    print("Phonological features have been written to phonological_features.txt")


'''
# print feature_categories outcome as below: 
feature_categories = extract_feature_categories(ipa_to_phonemefeats)
feature_categories = {
    'symbol_type': ['</q>', '</s>', '<s>', '<sil>', '<w>', 'phoneme'], 
    'vowel_consonant': ['consonant', 'vowel'], 
    'VUV': ['unvoiced', 'voiced'], 
    'vowel_frontness': ['back', 'central', 'central_back', 'front', 'front_central'], 
    'vowel_openness': ['close', 'close-mid', 'close_close-mid', 'mid', 'open', 'open-mid'], 
    'vowel_roundedness': ['rounded', 'unrounded'], 
    'stress': ['unstressed'], 
    'consonant_place': ['alveolar', 'bilabial', 'glottal', 'labial-velar', 'labiodental', 'palatal', 'postalveolar', 'uvular', 'velar'], 
    'consonant_manner': ['affricate', 'approximant', 'fricative', 'lateral-approximant', 'nasal', 'stop'], 
    'diacritic': ['epiglottal', 'syllabic'], 
    }
'''