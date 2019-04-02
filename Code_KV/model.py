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

def position_encoding(sentences_ize,embedding_size):
    """
    对问题的input加入位置编码PE
    :param sentence_size:
    :param embedding_size:
    :return:
    """
    encoding = np.ones((embedding_size,sentences_ize),dtype=np.float32)
    ls = sentences_ize +1
    le = embedding_size+1
    for i in range(1,le):
        for j in range(1,ls):
            encoding[i-1,j-1] = (i-(le-1)/2) * (j-(ls-1)/2)
    encoding = 1 + 4*encoding/embedding_size/sentences_ize
    return np.transpose(encoding)


class KeyValueMemNN(object):
    def __init__(self,sess,size,idx_size,entity_idx_size,learning_rate,l2_lamda=None):
        self.sess = sess
        self.size =size#maxlen
        self.vocab_size=idx_size
        self.count_entities =entity_idx_size
        self.name="KeyValueMemNN"
        flags = tf.app.flags
        self.encoding= tf.constant(position_encoding(self.size[QUESTION]+1,flags.FLAGS.embedding_size),name="encoding")
        self.build_inputs()
        self.build_params()
        logits =  self.build_model() #batch * count_entities

        #训练部分
        #self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.answer))
        _loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.answer))
        with tf.name_scope("loss"):
            if l2_lamda :
                #加入L2正则化
                vars = tf.trainable_variables()
                lossL2=tf.add_n([tf.nn.l2_loss(v) for v in vars])
                self.loss_op=_loss_op + l2_lamda*lossL2
            else :
                self.loss_op=_loss_op
        #这里可能涉及到答案有多个，但是预测只有一个的情况
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)
        self.predict_op = tf.argmax(logits,1,name='predict_op')#logits是最后的输出
       # with tf.name_scope("accuracy"):

        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

    def build_inputs(self):
        """
        喂给神经网络的输入
        :return:
        """
        flags=tf.app.flags
        batch_size = flags.FLAGS.batch_size
        self.question = tf.placeholder(tf.int32,[None,self.size[QUESTION]+1],name="question")
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

            #E = tf.Variable(initializer([self.vocab_size,embedding_size]),name='E')
            self.A=tf.Variable(initializer([self.vocab_size,embedding_size]),name='A') #vocab_size *embedding_size
            #self.B=tf.Variable(initializer([self.vocab_size+1,embedding_size]),name='B')
            self.B = tf.Variable(initializer([embedding_size, self.count_entities]), name='B')
            #加入bias
            self.bias =tf.Variable(tf.constant(0.1,tf.float32,[self.count_entities]),name='bias')
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
            q_0 = tf.reduce_sum(q_emb*self.encoding,1) # [batch_size,embedding]
            q = [q_0]
            for hop in range(hops):
                key_emb=tf.nn.embedding_lookup(self.A,self.keys)
                k = tf.reduce_sum(key_emb,2)#[batch_size,size_memory,2,embedding_size]

                #应用dropout 在key上
                one = tf.ones([memory_size,1],tf.float32)
                one_dropout = tf.nn.dropout(one,self.drop_memory,noise_shape=[memory_size,1])
                q_temp =tf.expand_dims(q[-1],-1) #[batch_size,embedding_size,1]
                q_temp =tf.transpose(q_temp,[0,2,1]) #[batch_size,1,embedding_size]
                #进行点乘，对应元素相乘
                product =k * q_temp # [batch_size,size_memory,embedding_size]

                dotted = tf.reduce_sum(product,2)#[batch_size,size_memory]
                probs = tf.nn.softmax(dotted) #得到输出的概率，即memory中各个slot的概率

                value_emb = tf.nn.embedding_lookup(self.A,self.values)#[batch_size,size_meomory,embedding_size]

                #应用dropout在value上
                value_emb_dropout= value_emb*one_dropout

                #消去size_mermory
                probs_temp = tf.transpose(tf.expand_dims(probs,-1),[0,2,1])#[batch_size,1,size_memory]
                v_temp = tf.transpose(value_emb_dropout,[0,2,1]) #[batch_size,embedding_size,size_memory]
                #v_temp * probs_temp ->[batch_size,embedding_size,size_memory]
                o = v_temp *probs_temp
                o_k = tf.reduce_sum(o,2) #[batch_size,embedding_size]

                R_k = self.R_list[hop]
                R_1 = self.R_list[0]
                q_k = tf.matmul(q[-1]+o_k,R_k)#[batch_size,embedding_size]
                q.append(q_k)
        # B:[embedding_size,entity_size]
        # qn_entity[batch_size,size_qn_entity]
        # -> y_emb[batch_size,size_qn_entity,embedding]
        #y_emb = tf.nn.embedding_lookup(self.B, self.qn_entities)
        #y = tf.reduce_sum(y_emb, 1)  # [batch_size,embedding_size]
        #return q_k*y#论文中，这里是B*y 然后和q_k进行内积，其中 y 是候选实体集合candidate
        #with tf.name_scope("output"):
         #   logits = tf.nn.dropout(tf.matmul(q_k,self.B),keep_prob=self.drop_memory)
        return tf.matmul(q_k,self.B) #[batch_size,entity_size]
    def batch_fit(self,batch_dict):
        flags = tf.app.flags
        dropout_memory = flags.FLAGS.dropout_memory
        feed_dict = {
            self.question:batch_dict[QUESTION],
            self.answer : batch_dict[ANSWER],
            self.qn_entities:batch_dict[QN_ENTITIES],
            self.keys:batch_dict[KEYS],
            self.values:batch_dict[VALUES],
            self.drop_memory:dropout_memory
        }
        #训练
        self.sess.run(self.optimizer,feed_dict=feed_dict)
        loss=self.sess.run(self.loss_op,feed_dict=feed_dict)
        return loss
    def predict(self,batch_dict):
        feed_dict = {
            self.question: batch_dict[QUESTION],
            self.answer: batch_dict[ANSWER],
            self.qn_entities: batch_dict[QN_ENTITIES],
            self.keys: batch_dict[KEYS],
            self.values: batch_dict[VALUES],
            self.drop_memory: 1.0
        }
        return self.sess.run(self.predict_op,feed_dict=feed_dict)
    def get_embedding_matrix(self):
        return self.sess.run(self.A)

