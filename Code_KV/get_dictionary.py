"""
    将问题，答案，实体，关系等
    映射成 word 级别的字典，原子级别，
    每一种类型（word级别和实体关系级别的，都有一个索引id对应
    比如，"Joseph Tom" 作为一个实体，有一个id索引对应，"Joseph" 和 "Tom" 单个单词各自有一个索引对应
"""
from collections import defaultdict
import random
import csv
from word_process_method import *
from tqdm import tqdm
words = set([]) #单词级别
entities = set([])
relations = set([])
all = set([])
top_k = 0 #记录答案的最多数目
def add_entity(entity):
    entities.add(entity)
    for w in entity.split(" "):
        words.add(clean_word(w))
def add_sentence(sentence):
    for w in sentence.split(" "):
        words.add(clean_word(w))
def read_kb_file( kb_path):
    with open(kb_path,'r',encoding='utf-8') as kb_file:
        print("reading kb_file...")
        reader = csv.DictReader(kb_file,delimiter="|",fieldnames=['subject','relation','object'])
        for row in tqdm(reader):
            e1,r,e2 = row['subject'],row['relation'],row['object']
            add_entity(e1)
            add_entity(e2)
            relations.add(r)
            relations.add("!_"+r)#将三元组的逆关系和加入进去
def read_doc_file(doc_path):
    with open(doc_path,'r',encoding='utf-8') as doc_file:
        print("reading doc file...")
        reader = csv.DictReader(doc_file,delimiter="|",fieldnames=['e','r','desc'])
        for row in tqdm(reader):
            ent,rel,desc = row['e'],row['r'],row['desc']
            add_entity(ent)
            relations.add(rel)
            relations.add("!_"+rel)
            add_sentence(desc)
def read_qa_file(qa_path):
    global top_k
    with open(qa_path,'r',encoding='utf-8') as qa_file:
        print("reading qa file ...")
        reader = csv.DictReader(qa_file,delimiter='\t',fieldnames=['q','a'])
        for row in tqdm(reader):
            q,a = row['q'],row['a']
            add_sentence(q)
            aa=a.split("|")
            top_k = max(top_k,len(aa))
            for e in aa:
                add_entity(e)
def write_idx(idx_path,s,write_num=None):
    print("writing ",idx_path," ...")
    #添加问题编号
    if write_num:
        for i in range(0,top_k):
            qa_num = "@{no}".format(no=i)
            s.add(qa_num)
    ordered_set = sorted(s)  # 排序
    id=1
    with open(idx_path,'w',newline='',encoding='utf-8') as idx_file:
        writer  = csv.DictWriter(idx_file,delimiter='\t',fieldnames=['x','count'])
        for x in ordered_set :
            writer.writerow({'x':x,'count':id})
            id = id+1
if __name__ == "__main__":
    dataset = "wiki"
    path = "../data/movieqa/"
    tran_path = path+"clean_{name}_qa_train.txt".format(name=dataset)
    test_path = path+"clean_{name}_qa_test.txt".format(name=dataset)
    dev_path = path+"clean_{name}_qa_dev.txt".format(name=dataset)
    kb_path = path+"clean_wiki_kb.txt" #.format(name=dataset)
    doc_path = path+"clean_wiki_doc.txt"
    read_doc_file(doc_path)
    read_kb_file(kb_path)
    read_qa_file(tran_path)
    read_qa_file(test_path)
    read_qa_file(dev_path)
    write_idx(path + "{name}_word_idx.txt".format(name=dataset),words)
    write_idx(path + "{name}_entity_idx.txt".format(name=dataset), entities)
    write_idx(path + "{name}_relation_idx.txt".format(name=dataset), relations)
    all = all.union(words,entities,relations)
    write_idx(path + "{name}_idx.txt".format(name=dataset), all,write_num=True)