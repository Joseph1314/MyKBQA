"""
    清洗QA数据集，将其变成格式容易处理的形式
"""
import csv
from word_process_method import *
input_qa = '../data/movieqa/movieqa/questions/wiki_entities/wiki-entities_qa_dev.txt'
input_entity = '../data/movieqa/clean_entities.txt'
output_qa = '../data/movieqa/clean_wiki_qa_dev.txt'

def main():
    valid_entites_set = read_file_as_set(input_entity)
    with open(input_qa,'r',encoding='utf-8') as in_qa_file:
        with open(output_qa,'w',newline='',encoding='utf-8') as out_qa_file:
            writer =  csv.DictWriter(out_qa_file,delimiter='\t',fieldnames=['question','answer'])
            for line in in_qa_file:
                line = clean_line(line)
                q,ans = line.split("?\t")
                q_words = q.split(" ")
                q_words = q_words[1:]#第一个是数字，需要去掉
                q_words = [clean_word(w) for w in q_words] #清洗数据
                ans_ent = ans.split(",")
                ans_ent = [clean_word(a) for a in ans_ent]

                valid_ans_entities = []
                has_invalid_word = False
                for word in ans_ent:
                    if word in valid_entites_set:
                        valid_ans_entities.append(word)
                    else :
                        has_invalid_word = True#有答案中的词不在实体集
                if has_invalid_word:
                    is_a_valid_split,valid_ans_entities = get_valid_entities(ans_ent,valid_entites_set,0)
                    valid_ans_entities = list(reversed(valid_ans_entities))

                #如果没有找到合法的答案实体，尽可能多的选择valid_entity_set中的实体
                if len(valid_ans_entities) ==0:
                    #print("empty")
                    for w in ans_ent:
                        if w in valid_entites_set:
                           #print(w,"find")
                            valid_ans_entities.append(w)

                if len(valid_ans_entities) >0:#按照合理的格式进行写入文件
                    valid_ans_entities.sort()
                    writer.writerow({'question':' '.join(q_words),'answer':'|'.join(valid_ans_entities)})

            print("answer_entity_size",len(valid_ans_entities))
            print("ok")

if __name__ == "__main__":
    #测试一下组合答案的效果
    """
    print(get_valid_entities(["monster in law", "they shoot horses", "don't they", "agnes of god"],
                             set(["monster in law", "they shoot horses don't they",
                                  "agnes", "agnes of god", "a", "agnes"]), 0))
    """
    main()
