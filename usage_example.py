# -*- coding:utf-8 -*-
# @Author: revised by RilaShu
# @DateTime: 11.03.2018

import gensim
import pandas as pd

embedding_file = './word2vec.txt'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=False)
print('Model loaded.')
print(pd.Series(word2vec_model.most_similar(u'乔峰')))