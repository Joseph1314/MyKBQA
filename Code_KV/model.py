"""
    经过12天的学习和处理数据
    今天开始编写模型kv记忆网
"""
import tensorflow as tf
import numpy as np
QUESTION = "question"
QN_ENTITIES = "qn_entities"
ANS_ENTITIES = "ans_entities"
SOURCES = "sources"
RELATIONS = "relations"
TARGETS = "targets"
ANSWER = "answer"
KEYS = "keys"
VALUES = "values"
class KeyValueMemNN(object):
    def __init__(self,sess,size,idx_size,entity_idx_size):
        self.sess = sess
        self.size =size
        self.vocab_size=idx_size
        self.count_entities =entity_idx_size
        self.name="KeyValueMemNN"
        self.build_inputs()
        self.build_params()
        logits =  self.build_model() #batch * count_entities

        #训练部分

    def build_inputs(self):
        """
        喂给神经网络的输入
        :return:
        """
        flags=tf.app.flags
        batch_size = flags.FLAGS.batch_size
        self.question = tf.placeholder(tf.int32,[None,self.size[QUESTION]],name="question")
        self.qn_entities =tf.placeholder(tf.int32,[None,self.size[QN_ENTITIES]],name="qn_entities")
        self.answer = tf.placeholder(tf.int32,shape=[None],name="answer")
        self.keys= tf.placeholder(tf.int32,[None,self.size[KEYS],2],name="keys")
        self.values = tf.placeholder(tf.int32,[None,self.size[VALUES]],name="values")
        self.drop_memory = tf.placeholder(tf.float32)
    def build_params(self):
        """
        神经网络中需要进行训练的变量
        :return:
        """
        flags =tf.app.flags
        embedding_size =flags.FLAGS.embedding_size
        hops = flags.FLAGS.hops
        with tf.variable_scope(self.name):
            #创建一个空的slot
            nil_word_slot = tf.constant(np.zeros([1,embedding_size]),dtype=tf.float32)
            initializer = tf.contrib.layers.xavier_initializer()

            E = tf.Variable(initializer([self.vocab_size,embedding_size]),name='E')
            self.A=tf.concat([nil_word_slot,E],axis=0) #vocab_size+1 *embedding_size
            self.B=tf.Variable(initializer([embedding_size,self.count_entities]),name='B')
            self.R_list=[]
            for k in range(hops):
                R_k = tf.Variable(initializer([embedding_size,embedding_size]),name='H')
                self.R_list.append(R_k)
    def build_model(self):
        flags =tf.app.flags
        hops=flags.FLAGS.hops
        batch_size = flags.FLAGS.batch_size
        memory_size= self.size[KEYS]
        with tf.variable_scope(self.name):

            # A:[vocab_size+1,embedding_size]
            # question[batch_size,size_question]
            # -> q_emb[batch_size,size_question,embedding]
            q_emb = tf.nn.embedding_lookup(self.A,self.question)
            q_0 = tf.reduce_sum(q_emb,1) # [batch_size,embedding]
            q = [q_0]
            for hop in range(hops):
                key_emb=tf.nn.embedding_lookup(self.A,self.keys)
                k = tf.reduce_sum(key_emb,2)
