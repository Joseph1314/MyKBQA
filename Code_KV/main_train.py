"""
    开始训练
"""
import argparse
import os
import numpy as np
import random
import tensorflow as tf
import tqdm
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from read_kv_data import DataReader
from read_kv_data import get_maxlen
from model import KeyValueMemNN
from word_process_method import *

flags =tf.app.flags
flags.DEFINE_float("learning_rate",0.005,"learning rate for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm",40.0,"Clip gradients to this norm.")
flags.DEFINE_integer("evaluation_interval",1000,"Evaluate and print results every x steps")
flags.DEFINE_integer("batch_size",10,"Batch size for training.")
flags.DEFINE_integer("hops",2,"Number of hops in KVMemNN")
flags.DEFINE_integer("epochs",5,"Number of epochs to train for")
flags.DEFINE_integer("embedding_size",200,"Embedding size for embedding matrix")
flags.DEFINE_float("dropout_memory",1,"keep probability for keeping the memory slots")
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

path="../data/movieqa/"
encoder ={}
def get_single_column_from_batch(batch_examples,maxlen,col_name,test_data=False):
    """
    得到一批样本数据
    :param batch_examples:
    :param maxlen:
    :param col_name:
    :return:
    """
    global encoder
    batch_size = FLAGS.batch_size
    column=[]
    for i in range(batch_size):
        num_ans = len(batch_examples[i][ANS_ENTITIES])
        example = pad(batch_examples[i][col_name],maxlen[col_name])
        #防止答案有多个，将一对多的问题-答案转换成一对一
        """
            例如一条example Q-> a1,a2,a3
            转化成三条样例
            Q->a1
            Q->a2
            Q->a3
        """
        #example=np.array(example)
        tmp_example = [0]
        for j in range(num_ans):
            num = "@{no}".format(no=j)
            tmp_example = example[:]
            if col_name == QUESTION and test_data == False:
                tmp_example.append(encoder[num]-1)
            column.append(np.array(tmp_example))
    #print("problems:",len(column))
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

        if maxlen[KEYS] >(example_length):
            src = pad(src,maxlen[KEYS])
            rel = pad(rel,maxlen[KEYS])
            tar = pad(tar,maxlen[KEYS])
            example_indice_to_pick=range(len(src))
        else :
            example_indice_to_pick = random.sample(range(example_length),maxlen[KEYS])

        for index in example_indice_to_pick:
            memories_keys.append(np.array([src[index],rel[index]]))
            memories_values.append(tar[index])
         #一条QA数据集中的相关三元组已经组成了k-v形式
        # 防止答案有多个，将一对多的问题-答案转换成一对一 需要重复三元组
            """
                例如一条example Q-> a1,a2,a3
                转化成三条样例
                Q->a1
                Q->a2
                Q->a3
            """
        num_ans = len(batch_examples[i][ANS_ENTITIES])
        for j in range(num_ans):
            col_key.append(np.array(memories_keys))
            col_value.append(np.array(memories_values))
    return np.array(col_key),np.array(col_value)
def prepare_batch(batch_example,maxlen,test_data=False):
    batch_size =FLAGS.batch_size
    batch_dict={}
    batch_dict[QUESTION]=get_single_column_from_batch(batch_example,maxlen,QUESTION)
    batch_dict[QN_ENTITIES]=get_single_column_from_batch(batch_example,maxlen,QN_ENTITIES,test_data=test_data)

    #这三条可能没有用，之后删
    batch_dict[SOURCES]=get_single_column_from_batch(batch_example,maxlen,SOURCES)
    batch_dict[RELATIONS]=get_single_column_from_batch(batch_example,maxlen,RELATIONS)
    batch_dict[TARGETS]=get_single_column_from_batch(batch_example,maxlen,TARGETS)

    batch_dict[KEYS],batch_dict[VALUES]=get_kv_from_batch(batch_example,maxlen)

    #label answer
    labels =[]
    for i in range(batch_size):
        """
        tmp_ans = batch_example[i][ANS_ENTITIES][:]
        tmp_ans.sort()
        for ans in tmp_ans:
            ans2arr = [0] * entity_size
            ans2arr[ans] = 1
            #print("size:",len(ans2arr))
            labels.append(np.array(ans2arr))
        """
        for ans in batch_example[i][ANS_ENTITIES]:
            labels.append(ans)


    #labels.sort()#这里好像排序之后就不能收敛了
    batch_dict[ANSWER]=np.array(labels)
    return batch_dict

def save_model(sess):
    #保存模型
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    save_path = saver.save(sess,os.path.join(FLAGS.checkpoint_dir,"model_kv.ckpt"))
    print("Model saved in file %s"%save_path)
def get_accuracy(model,examples,maxlen):
    """
    得到正确率
    :param model:
    :param examples:
    :param maxlen:
    :return:
    """
    batch_size =FLAGS.batch_size
    num_example =len(examples)
    batches = zip(range(0,num_example-batch_size),range(batch_size,num_example))
    batches = [(start,end) for start,end in batches]

    count_total=0.0
    count_correct = 0.0
    id=1
    for start, end in batches:
        # print("start,end")
        if id > 100:
            break
        id = id + 1
        batch_examples = examples[start:end]
        batch_dict = prepare_batch(batch_examples, maxlen)
        prediction = model.predict(batch_dict)
        for i in range(len(batch_examples)):
            correct_answer = set(batch_examples[i][ANS_ENTITIES])  # 答案是一个集合
            if prediction[i] in correct_answer:
                count_correct = count_correct + 1
            count_total = count_total + 1
            print(float(count_correct / count_total))
    return float(count_correct / count_total) if count_total != 0 else 0

def main(args):
    global encoder
    max_slots =FLAGS.max_slots
    encoder=read_file_as_dict(args.idx)
    maxlen = get_maxlen(args.train_examples,args.test_examples)
    maxlen[KEYS]=min(maxlen[SOURCES],max_slots)
    maxlen[VALUES]=min(maxlen[SOURCES],max_slots)
    assert (maxlen[KEYS]==maxlen[VALUES])
    args.input_examples = args.train_examples
    train_reader = DataReader(args, maxlen, share_idx=True,data_name="Train_example")
    train_examples = train_reader.get_examples()
    entity_size =train_reader.get_entities_size()
    args.input_examples = args.test_examples
    test_reader = DataReader(args, maxlen, share_idx=True,data_name="Test_example")
    test_examples = test_reader.get_examples()

    #args.input_examples = args.dev_examples
    #dev_reader = DataReader(args, maxlen, share_idx=True)
    #dev_examples = dev_reader.get_examples()

    num_train = len(train_examples)
    batch_size = FLAGS.batch_size
    batches = zip(range(0,num_train-batch_size),range(batch_size,num_train))
    batches = [(start,end) for start,end in batches]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = KeyValueMemNN(sess,maxlen,train_reader.get_idx_size(),train_reader.get_entities_size(),learning_rate=FLAGS.learning_rate)
        if os.path.exists(os.path.join(FLAGS.checkpoint_dir,"model_kv.ckpt")):
            saver = tf.train.Saver()
            save_path = os.path.join(FLAGS.checkpoint_dir,"model_kv.ckpt")
            saver.restore(sess,save_path)
            print("Model restored from file %s" % save_path)
        max_test_accuracy = -1.0
        for epoch in range(1,FLAGS.epochs+1):
            np.random.shuffle(batches)
            step=1
            for start,end in batches:
                batch_examples = train_examples[start:end]
                batch_dict = prepare_batch(batch_examples,maxlen,entity_size)
                loss = model.batch_fit(batch_dict)
                predictions = model.predict(batch_dict)
                labels = tf.constant(batch_dict[ANSWER],tf.int64)
                train_accuracy = tf.contrib.metrics.accuracy(predictions, labels)
                #correct_predictions=tf.equal(predictions,tf.argmax(labels, 1))
                #train_accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")
                print(step,":EPOCH={epoch}:BATCH_TRAIN_LOSS={class_loss}:BATCH_TRAIN_ACC:{train_acc}". \
                      format(epoch=epoch, class_loss=loss, train_acc=sess.run(train_accuracy)))
                step =step+1
                if epoch > 0 and step % FLAGS.evaluation_interval == 0:
                    test_accuracy = get_accuracy(model, test_examples, maxlen)
                    #train_accuracy = get_accuracy(model, train_examples, maxlen)
                    if test_accuracy > max_test_accuracy:
                        save_model(sess)
                        max_test_accuracy = test_accuracy
                    print("Evaluate:EPOCH={epoch}:TEST_ACCURACY={test_accuracy}:BEST_ACC={best_acc}". \
                          format(epoch=epoch, test_accuracy=(test_accuracy),best_acc=max_test_accuracy))
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_examples', default=path+"wiki_small_qa_train.txt")
    parser.add_argument('--test_examples', default=path+"wiki_kv_qa_test.txt")
    #parser.add_argument('--dev_examples', help='the dev file', required=True)
    parser.add_argument('--word_idx', help='word vocabulary', default=path+"wiki_word_idx.txt")
    parser.add_argument('--entity_idx', help='entity vocabulary', default=path+"wiki_entity_idx.txt")
    parser.add_argument('--relation_idx', help='relation vocabulary', default=path+"wiki_relation_idx.txt")
    parser.add_argument('--idx', help='overall vocabulary', default=path+"wiki_idx.txt")
    args = parser.parse_args()
    main(args)