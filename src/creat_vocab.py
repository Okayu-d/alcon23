import os
import sys
import json
import numpy as np

import pandas as pd
from vocab import Vocab

def main():
    print('load annotations.csv')
    train_df = pd.read_csv('./alcon23/input/dataset/train/annotations.csv').set_index('ID')
    print('load https://www.utf8-chartable.de/unicode-utf8-table.pl?start=12352&number=128&view=3')
    unicode_table=pd.read_html("https://www.utf8-chartable.de/unicode-utf8-table.pl?start=12352&number=128&view=3")[1]
    
    print('make index <---> Unicode')
    v = Vocab()
    for index, row in train_df.iterrows():
        for i in range(1, 4):
            key = f'Unicode{i}'
            v.word2index(row[key], train=True)
    index2word = {}
    word2index = {}
    for i, u in enumerate(v.to_dict()['index2word']):
        index2word[i] = u
        word2index[u] = i

    print('make Unicode <---> character')
    unicode2char =  {}
    char2unicode = {}
    for uni in v.to_dict()['index2word']:
        if (uni is np.nan): break # 追加 NaN(float)の時は飛ばす．（dict()の最後がNaN)
        ch = unicode_table[unicode_table['Unicodecode point'] == uni]['character'].item()
        unicode2char[uni] = ch
        char2unicode[ch] = uni
    
    rarity = {}
    for i, key in enumerate(list(dict(sorted(v.to_dict()['counts'].items(), key=lambda x: -x[1])).keys())):
        rarity[key] = i   

    os.makedirs('./alcon23/input/vocab', exist_ok=True)
    with open('./alcon23/input/vocab/index2unicode.json', 'w') as f:
#     json.dump(index2word, f, indent=4)
        json.dump(v.to_dict()['index2word'], f, indent=4)

    with open('./alcon23/input/vocab/unicode2index.json', 'w') as f:
        json.dump(word2index, f, indent=4)
    with open('./alcon23/input/vocab/char2unicode.json', 'w') as f:
        json.dump(char2unicode, f, indent=4)
    with open('./alcon23/input/vocab/unicode2char.json', 'w') as f:
        json.dump(unicode2char, f, indent=4)
    with open('./alcon23/input/vocab/rarity.json', 'w') as f:
        json.dump(rarity, f, indent=4)

if __name__ == '__main__':
    main()
