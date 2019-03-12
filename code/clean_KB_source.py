"""

    用来处理原始的KB文件，将其转换为能容易切割识别的格式
    例如原来的格式是 kkk vv rr_by bbb
    关系是rr_by 但是实体是(kkk vv) 和(bbb) 两个，(kkk vv) 含有一个空格，
    不能直接用空格进行正则化的切割，需要通过一些算法将其转换为容易处理的格式
    :-> kkk vv |rr_by| bbb 中间用|进行分割容易处理

"""
import csv
import argparse
import unicodedata
from word_process_method import *
def main(args):
    relation_set =set([])
    entity_set = set([])
    valid_entity_set = read_file_as_set(args.entities)
    with open( args.kb_source,'r') as kb_file:
        #进行读写
        with open(args.out_kb,'w',newline='') as out_kb_file:
            #out_kb_file是对原文件进行转换格式之后的文件
            with open(args.out_doc,'w',newline='') as out_doc_file:
                #存储的是三元组(subject,relation,object)
                #可以看做是doc文档类型的kb
                kb_writer = csv.DictWriter(out_kb_file,delimiter='|',
                                           fieldnames=['subject','relation','object'])
                doc_writer = csv.DictWriter(out_doc_file,delimiter='|',
                                            fieldnames=['entity','fieldname','content'])
                for id,line in enumerate(kb_file):#对文件进行逐行读取
                    #先不进行 行清洗
                    line = clean_line(line)
                    #print(line)
                    #print(id)
                    if(len(line)) == 0:
                        continue
                    e1,e2,r = None,None,None
                    cur = [] #保存当前遇到的单词，后面会连接成一个实体
                    found_relation = False
                    for word in line.split(" ")[1:]:
                        if '_' in word and not found_relation:
                            #第一次遇见'_'符号，默认只有在relation中才能出现 '_' 符号
                            r = word
                            relation_set.add(r)
                            e1 = " ".join(cur)
                            cur = []#当前列表清空'
                            found_relation = True
                        else :
                            cur.append(word)
                    e2 = " ".join(cur) #将列表元素用空格连接成一个大的整体元素
                    #先不进行单词的清洗
                    #e1 = clean_word(e1)
                    #e2 = clean_word(e2)
                    if r == "has_plot":
                        write_doc(e1,e2,r,valid_entity_set,doc_writer)
                    else :
                        write_kb(e1,e2,r,valid_entity_set,kb_writer,entity_set)#会在写的过程中补充实体，因为答案answer可能包含多个实体
    print("number of entity",len(entity_set))
    print("number of relation",len(relation_set))
def write_kb(e1,e2s,relation,valid_entity_set,kb_writer,entity_set):
    """
    将三元组写入文件
    :param e1:
    :param e2s:
    :param relation:
    :param valid_entity_set:
    :param kb_writer:
    :param entity_set:
    :return:
    """
    clean_word(e1)
    entity_set.add(e1)
    #print(e2s)
    for e2 in e2s.split(","):#答案的格式是多个答案中间用逗号隔开 如果不切分，默认答案是多个
        # 暂时先不清理word
        e2= clean_word(e2)
        entity_set.add(e2)
        #print("ok?")
        #if e1 in valid_entity_set and e2 in valid_entity_set:
        if True:
            dict ={'subject':e1,'relation':relation,'object':e2}
            #print("write one")
            kb_writer.writerow(dict)
def write_doc(e1,e2s,relation,valid_entity_set,doc_writer):
    """
    文档类的三元组写入文件
    :param e1:
    :param e2s:
    :param relation:
    :param valid_entity_set:
    :param doc_writer:
    :return:
    """
    #if e1 in valid_entity_set :
    if True:
        dict = {'entity': e1, 'content': e2s, 'fieldname': relation}#先不对e2s进行清洗 clean_word(e2s)
        doc_writer.writerow(dict)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='具体化各个参数')
    parser.add_argument('--kb_source',help='the raw kb_file',default='../data/movieqa/full_kb.txt')
    parser.add_argument('--entities',help='the full entities in kb',default='../data/movieqa/entities.txt')
    parser.add_argument('--out_kb',help='the processed kb_file',default='../data/movieqa/ac_kb.txt')
    parser.add_argument('--out_doc',help='the processed doc file',default='../data/movieqa/ac_doc.txt')
    args=parser.parse_args()
    main(args)

