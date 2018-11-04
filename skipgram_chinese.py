# -*- coding:utf-8 -*-
# @Author: revised by RilaShu
# @DateTime: 11.02.2018

import collections
import math
import random
import jieba
import numpy as np
from six.moves import xrange
import tensorflow as tf
import os

# Step 1: Download the data.
# Read the data into a list of strings.
def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    # 读取停用词
    stop_words = []
    with open('stop_words.txt', "r", encoding="UTF-8") as fStopWords:
        line = fStopWords.readline()
        while line:
            stop_words.append(line[:-1]) # 去\n
            line = fStopWords.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，去除停用词，得到词典
    sFolderPath = 'JinYong\'s Works'
    lsFiles = []
    for root, dirs, files in os.walk(sFolderPath):
        for file in files:
            if file.endswith(".txt"):
                lsFiles.append(os.path.join(root, file))
    raw_word_list = []
    for item in lsFiles:
        with open(item, "r", encoding='UTF-8') as f:
            line = f.readline()
            while line:
                while '\n' in line:
                    line = line.replace('\n', '')
                while ' ' in line:
                    line = line.replace(' ', '')
                # 如果句子非空
                if len(line) > 0:
                    raw_words = list(jieba.cut(line, cut_all=False))
                    for item in raw_words:
                        # 去除停用词
                        if item not in stop_words:
                            raw_word_list.append(item)
                line = f.readline()
    return raw_word_list

words = read_data()
print('Data size', len(words))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 100000
def build_dataset(words):
    # 词汇编码
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count", len(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 使用生产的词汇编码将前面产生的 string list[words] 转变成 num list[data]
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # 反转字典 key为词汇编码 values为词汇本身
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
#删除words节省内存
del words  
print('Most common words ', count[1:6])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    """
    就是对于一个中心词 在window范围 随机选取 num_skips个词，产生一系列的
    ( one of left or right words in window, center) 作为(batch_instance, label)
    :param batch_size: batch size
    :param num_skips: 产生label的次数限制
    :param skip_window: 窗口大小
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)#(1, batch_size)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)# (batch_size, 1)
    span = 2 * skip_window + 1  # [ left target right ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data) # ?
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
# 显示示例
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 100
skip_window = 1       
num_skips = 2         
valid_size = 4      #切记这个数字要和len(valid_word)对应，要不然会报错哦
valid_window = 100  
num_sampled = 64    # Number of negative examples to sample.

#验证集
valid_word = ['说', '实力', '害怕', '少林寺']
valid_examples = [dictionary[li] for li in valid_word]

graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 权重矩阵（也就是要被学习到的）
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # 选取张量embeddings中对应train_inputs索引的值
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 转化变量输入，适配NCE
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases, 
                                         inputs=embed, 
                                         labels=train_labels,
                                         num_sampled=num_sampled, 
                                         num_classes=vocabulary_size))

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 使用所学习的词向量来计算一个给定的 minibatch 与所有单词之间的相似度（余弦距离）
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 5000000

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[:top_k]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# Step 6: 输出词向量
with open('word2vec.txt', "w", encoding="UTF-8") as fW2V:
    fW2V.write(str(vocabulary_size) + ' ' + str(embedding_size) + '\n')
    for i in xrange(final_embeddings.shape[0]):
        sWord = reverse_dictionary[i]
        sVector = ''
        for j in xrange(final_embeddings.shape[1]):
            sVector = sVector + ' ' + str(final_embeddings[i, j])
        fW2V.write(sWord + sVector + '\n')

