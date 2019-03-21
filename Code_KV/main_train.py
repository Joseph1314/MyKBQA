"""
    开始训练
"""
import argparse
import os
import numpy as np
import random
import tensorflow as tf
import tqdm

from read_kv_data import DataReader
from read_kv_data import get_maxlen
from model import KeyValueMemNN
from word_process_method import *

flags =tf.app.flags
flags.DEFINE_float("learning_rate",0.01,"learning rate for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm",40.0,"Clip gradients to this norm.")
flags.DEFINE_integer("evaluation_interval",5,"Evaluate and print results every x epochs")
flags.DEFINE_integer("batch_size",8,"Batch size for training.")
flags.DEFINE_integer("hops",2,"Number of hops in KVMemNN")
flags.DEFINE_integer("epochs",1000,"Number of epochs to train for")
flags.DEFINE_integer("embedding_size",128,"Embedding size for embedding matrix")
flags.DEFINE_float("dropout_memory",1.0,"keep probability for keeping the memory slots")
flags.DEFINE_string("checkpoint_dir","checkpoints","checkpoint directory for the model saved")
flags.DEFINE_integer("max_slots",64,"maximum slots in the memory")

FLAGS =flags.FLAGS
QUESTION = "question"
QN_ENTITIES = "qn_entities"
ANS_ENTITIES = "ans_entities"
SOURCES = "sources"
RELATIONS = "relations"
TARGETS = "targets"
ANSWER = "answer"
KEYS = "keys"
VALUES = "values"

def get_single_column_from_batch(batch_examples,maxlen,col_name):
    """
    得到一批样本数据
    :param batch_examples:
    :param maxlen:
    :param col_name:
    :return:
    """
    batch_size = FLAGS.batch_size
    column=[]
    for i in range(batch_size):
        num_ans = len(batch_examples[i][ANSWER])
        example = pad(batch_examples[i][col_name],maxlen[col_name])
        #防止答案有多个，将一对多的问题-答案转换成一对一
        """
            例如一条example Q-> a1,a2,a3
            转化成三条样例
            Q->a1
            Q->a2
            Q->a3
        """
        for j in range(num_ans):
            column.append(np.array(example))
    return np.array(column)
def get_kv_from_batch(batch_examples,maxlen):
    """
    将一条QA数据中涉及到的三元组变成k-v形式，因为要防止多个答案的问题，需要复制相同答案数目的k-v数据对
    :param batch_examples:
    :param maxlen:
    :return:
    """
    batch_size =FLAGS.batch_size
    col_key=[]
    col_value=[]
    for i in range(batch_size):
        assert(len(batch_examples[i][SOURCES]) ==len(batch_examples[i][RELATIONS]))
        assert(len(batch_examples[i][SOURCES]) ==len(batch_examples[i][TARGETS]))
        example_length = len(batch_examples[i][SOURCES])
        memories_keys = []
        memories_values = []
        src = batch_examples[i][SOURCES]
        rel = batch_examples[i][RELATIONS]
        tar = batch_examples[i][TARGETS]

        if maxlen[KEYS] > len(example_length):
            src = pad(src,maxlen[KEYS])
            rel = pad(rel,maxlen[KEYS])
            tar = pad(tar,maxlen[KEYS])
            example_indice_to_pick=range(len(src))
        else :
            example_indice_to_pick = random.sample(range(example_length),maxlen[KEYS])

        for index in example_indice_to_pick:
            memories_keys.append(np.array([src[index],rel[index]]))
            memories_values.append(np.array([tar[index]]))
         #一条QA数据集中的相关三元组已经组成了k-v形式
        # 防止答案有多个，将一对多的问题-答案转换成一对一 需要重复三元组
            """
                例如一条example Q-> a1,a2,a3
                转化成三条样例
                Q->a1
                Q->a2
                Q->a3
            """
        num_ans = len(batch_examples[i][ANSWER])
        for j in range(num_ans):
            col_key.append(memories_keys)
            col_value.append(memories_values)
    return np.array(col_key),np.array(col_value)
def prepare_batch(batch_example,maxlen):
    batch_size =FLAGS.batch_size
    batch_dict={}
    batch_dict[QUESTION]=get_single_column_from_batch(batch_example,maxlen,QUESTION)
    batch_dict[QN_ENTITIES]=get_single_column_from_batch(batch_example,maxlen,QN_ENTITIES)

    #这三条可能没有用，之后删
    batch_dict[SOURCES]=get_single_column_from_batch(batch_example,maxlen,SOURCES)
    batch_dict[RELATIONS]=get_single_column_from_batch(batch_example,maxlen,RELATIONS)
    batch_dict[TARGETS]=get_single_column_from_batch(batch_example,maxlen,TARGETS)

    batch_dict[KEYS],batch_dict[VALUES]=get_kv_from_batch(batch_example,maxlen)

    #label answer
    labels =[]
    for i in range(batch_size):
        for ans in batch_example[i][ANSWER]:
            labels.append(ans)
    batch_dict[ANSWER]=np.array(labels)
    return batch_dict

def save_model(sess):
    #保存模型
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    save_path = saver.save(sess,os.path.join(FLAGS.checkpoint_dir,"model_kv.ckpt"))
    print("Model saved in file %s"%save_path)
def main(args):
    max_slots =FLAGS.max_slots
    maxlen = get_maxlen(args.train_examples,args.test_examples,args.dev_examples)
    maxlen[KEYS]=min(maxlen[SOURCES],max_slots)
    maxlen[VALUES]=min(maxlen[SOURCES],max_slots)
    assert (maxlen[KEYS]==maxlen[VALUES])
    args.input_examples = args.train_examples
    train_reader = DataReader(args, maxlen, share_idx=True)
    train_examples = train_reader.get_examples()

    args.input_examples = args.test_examples
    test_reader = DataReader(args, maxlen, share_idx=True)
    test_examples = test_reader.get_examples()

    args.input_examples = args.dev_examples
    dev_reader = DataReader(args, maxlen, share_idx=True)
    dev_examples = dev_reader.get_examples()