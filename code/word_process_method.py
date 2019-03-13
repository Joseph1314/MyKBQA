"""
    清洗数据和读取文件等的一些方法
"""
import unicodedata
import chardet
def clean_word(word):
    """
    清洗单词，并且将单词同意转化为小写
    :param word:
    :return:
    """
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
if __name__ == '__main__':
    w=clean_word('déjà')
    print(w)