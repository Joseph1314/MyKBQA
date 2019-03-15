"""
    处理实体，将实体全部转换为小写,清理相关的音符文件
    移除逗号，并且按照字典序进行排序
"""
from word_process_method import *
from sortedcontainers import SortedSet#排序的集合
input='../data/movieqa/entities.txt'
output = '../data/movieqa/clean_entities.txt'
def main():
    NEWLINE = "\n"
    ent_set = SortedSet([])
    cnt_raw = 0
    cnt_processes = 0
    with open(input,'r',encoding='utf-8') as entity_file:
        with open(output,'w',encoding='utf-8') as cleaned_entity_file:
            for entity in (entity_file):
                #if entity[0]=='R' and entity[1]=='e':
                #    print(entity)
                cnt_raw += 1
                flag=False
                cleaned_ent = clean_word(entity)
                if len(cleaned_ent) >0:
                    ent_set.add(cleaned_ent)
            #将排好序并且清洗好的entity写入文件
            for entity in ent_set:
                cnt_processes += 1
                cleaned_entity_file.write(entity+NEWLINE)
    print("cnt_raw",cnt_raw)
    print("cnt_processed",cnt_processes)
if __name__ == '__main__':
    main()
