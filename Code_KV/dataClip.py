"""
    划分qa数据集，将训练数据缩小，写入新的文件
"""
import csv
import random
from tqdm import tqdm
input_path="../data/movieqa/wiki_kv_qa_train.txt"
output_path="../data/movieqa/wiki_small_qa_train.txt"
QUESTION = "question"
QN_ENTITIES = "qn_entities"
ANS_ENTITIES = "ans_entities"
SOURCES = "sources"
RELATIONS = "relations"
TARGETS = "targets"
ANSWER = "answer"
KEYS = "keys"
VALUES = "values"

def partition_data(input_path,output_path,percentage):
    """
    抽取数据集中的子集，写入新的文件
    :param input_path:
    :param output_path:
    :return:
    """
    all_data=[]
    sub_data=[]
    fields = ['question', 'qn_entities', 'ans_entities', 'sources', 'relations', 'targets']
    with open(input_path,'r') as in_file:
        reader = csv.DictReader(in_file,delimiter='\t',fieldnames=fields)
        for row in tqdm(reader):
            col ={}
            col[QUESTION]=row[QUESTION]
            col[QN_ENTITIES]=row[QN_ENTITIES]
            col[ANS_ENTITIES]=row[ANS_ENTITIES]
            col[SOURCES]=row[SOURCES]
            col[RELATIONS]=row[RELATIONS]
            col[TARGETS]=row[TARGETS]
            all_data.append(col)
    all_len=len(all_data)
    sub_data= random.sample(all_data,int(all_len*percentage))
    with open(output_path,'w',newline='') as out_file:
        writer = csv.DictWriter(out_file,delimiter='\t',fieldnames=fields)
        for line in sub_data:
            out_row={
                QUESTION:line[QUESTION],
                QN_ENTITIES:line[QN_ENTITIES],
                ANS_ENTITIES:line[ANS_ENTITIES],
                SOURCES:line[SOURCES],
                RELATIONS:line[RELATIONS],
                TARGETS:line[TARGETS]
            }
            writer.writerow(out_row)
if __name__ =="__main__":
    partition_data(input_path,output_path,0.08)