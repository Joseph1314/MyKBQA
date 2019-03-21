"""
    清洗数据和读取文件等的一些方法
"""
import unicodedata
import csv
import chardet
from tqdm import tqdm
def clean_word(word):
    """
    清洗单词，并且将单词同意转化为小写
    :param word:
    :return:
    """
    if type(word) == bytes:
        word = word.decode('utf-8')
    word = word.strip('\n')
    word = word.strip('\r')
    word = word.lower()
    word = word.replace('%', '')  # 99 and 44/100% dead
    word = word.strip()
    word = word.replace(',', '')
    word = word.replace('.', '')
    word = word.replace('"', '')
    word = word.replace('\'', '')
    word = word.replace('?', '')
    word = word.replace('|', '')
    #word = word.replace(' ','')
    # print(type(word))
    word = word.encode("utf-8")  # Convert str -> unicode (Remember default encoding is ascii in python)
    word = word.decode("utf-8")
    word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('ascii')  # Convert normalized unicode to python str
    word = word.lower()  # Don't remove this line, lowercase after the unicode normalization
    word = word.encode("utf-8")  # Convert str -> unicode (Remember default encoding is ascii in python)
    word = word.decode("utf-8")
    return word
def clean_line(line):
    """
    对读取的一行进行清洗
    :param line:
    :return:
    """
    line = line.strip('\n')
    line = line.strip('\r')
    line = line.strip()
    line = line.lower()
    return line
def read_file_as_set(input_path):
    s = set()
    with open(input_path) as input_file:
        #first=True
        for line in input_file:
            line = (line.split('\n')[0])
            s.add(line)
    return  s
def read_file_as_dict(input_path):
    """
    读取文件为字典格式
    :param dictionary:
    :return:
    """
    dict = {}
    with open(input_path,'r',encoding="utf-8") as input_file:
        reader = csv.DictReader(input_path,delimiter = '\t',fieldnames=['col1','col2'])
        for row in tqdm(reader):
            if row['col1'] != None and row['col2']!=None:
                dict[row['col1']] = int(row['col2'])
    return dict
def get_valid_entities(potiential_ent_set,dictionary,pos):
    if pos >= len(potiential_ent_set):
        return  True,[]
    for i in range(pos,len(potiential_ent_set)):
        # 对于答案实体进行一个重新组合，来发现是不是存在另外的实体
        subSequence = " ".join(potiential_ent_set[pos:i+1])
        if subSequence in dictionary:
            is_a_valid_split,Seq = get_valid_entities(potiential_ent_set,dictionary,i+1)
            if is_a_valid_split:
                Seq.append(subSequence)
                return True,Seq
    return False,[]
def get_str_of_nested_seq(paths):
    result = []
    #print(paths)
    for p in paths:
        print("++++")
        print(p)
        result.append(",".join(p))
        print(result)
    #print(result)
    return "|".join(result)
def extract_dimension_from_tuples_as_list(list_of_tuples,dim):
    """
    从元组中提取相应的维度，即从三元组中提取s,r,t的其中之一
    :param list_of_tuples:
    :param dim:
    :return:
    """
    result=[]
    for t in list_of_tuples:
        result.append(t[dim])
    return result
def pad(arr,Len):
    """
    补全
    :param arr:
    :param len:
    :return:
    """
    copy_arr = list(arr)
    if len(copy_arr) <Len:
        while(len(copy_arr)<Len):
            copy_arr.append(0)
    return copy_arr
if __name__ == '__main__':
    w=clean_word('déjà')
    print(w)