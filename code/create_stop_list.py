"""

    生成stop words 的列表,根据单词级别word 的出现的频率frequency来决定，
    同时生成2个单词组成的短语作为补充 比如 described by 是described 和 by的组合

"""
import csv

import argparse
from sortedcontainers import SortedSet
from tqdm import tqdm
Frequency_threshold = 500 #判定为stop word 的频率阈值
def get_bigram(word_list):
    """
    生成一个句子中的2个词的组合的列表
    :param word_list:
    :return:
    """
    bigram = []
    for i in range(0,len(word_list)-1):
        bigram.append(word_list[i]+" "+word_list[i+1])
    return bigram;
def main(args):
    dict = {} # word -> freuency 的一个字典
    stop_list = SortedSet([])
    #这里需要一个循环，来处理三个train ,test dev,的qa_path
    with open(args.qa_path,'r',encoding='utf-8') as qa_file:
        reader = csv.DictReader(qa_file,delimiter ='\t',
                                fieldnames=['question','answer'])
        for row in tqdm(reader):
            q_words = row['question']#,row['answer'] 只是提取问题中的stop word
            q_words = q_words.split(" ")
            for w in q_words:
                freq = dict.get(w,0);
                dict[w]=freq+1
            for w in get_bigram(q_words):
                freq = dict.get(w,0)
                dict[w]=freq+1
    with open(args.doc_path,'r',encoding='utf-8') as doc_file:
        reader = csv.DictReader(doc_file,delimiter='|',fieldnames=['e','fieldname','content'])
        for row in tqdm(reader):
            content_words = row['content']
            content_words = content_words.split(" ")
            for w in content_words:
                freq = dict.get(w,0)
                dict[w] = freq+1
    #写入文件，stop_list
    with open(args.stop_list,'w',newline='',encoding='utf-8') as stop_file:
        writer = csv.DictWriter(stop_file,delimiter='\t',fieldnames=['word','count'])
        for w in dict.keys():
            if dict[w] >Frequency_threshold:
                stop_list.add(w)
        for w in stop_list:
            writer.writerow({'word':w,'count':dict[w]})
if __name__ == "__main__":
    path="../data/moiveqa/"
    parser = argparse.ArgumentParser(description='具体化参数')
    parser.add_argument('--qa_path',default=path+"clean_qa_full_xxx.txt")
    parser.add_argument('--doc_path',default=path+"ac_doc.txt")
    parser.add_argument('--stop_list',default=path+"stop_list.txt")
    args = parser.parse_args()
    main(args)