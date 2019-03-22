"""
    生成kv-data
"""
import argparse
import csv
import random
from word_process_method import *
from Knowledge_Graph import KnowledgeGraph
from search_Index import SearchEng
from parse_question import QuestionParser
from tqdm import tqdm
absPath="../data/movieqa/"
Max_Relevant_Entities = 4
Hops_From_Qn_Entities = 1
Max_Candidate_Entities = 1024
Max_Candidate_Tuples = 2048

def remove_high_degree_qn_entities(qn_entities):
    """
    移除问题实体中的高度节点，但是要至少保留一个
    ！！！
    注意，这里可能会存在一种情况，就是实体既不是高度节点，又不是kb图结构中的合理实体
    ！！！
    :param qn_entities:
    :return:
    """
    qn_clean_entities = set([])
    if len(qn_entities) > 1:
        for q in qn_entities:
            if q not in knowledge_base.get_high_degree_entities():
                qn_clean_entities.add(q)
    return qn_clean_entities if len(qn_clean_entities)>0 else qn_entities
def remove_invalid_ans_entities(ans_entities):
    """
    移除答案中，不合理的，即在构建的图结构中不存在的实体
    :param ans_entities:
    :return:
    """
    ans_clean_entities = set([])
    if len(ans_entities)>1:
        for a in ans_entities:
            if a in knowledge_base.get_entities():
                ans_clean_entities.add(a)
    return ans_clean_entities if len(ans_clean_entities) >0 else ans_entities
def get_neighboring_entities(entities,num_hops=2):
    """
    得到entities集合中所有实体的邻接节点的集合
    :param entities:
    :param num_hops:
    :return:
    """
    nbr_entities = set([])
    for ent in entities:
        for nbr in knowledge_base.get_candidate_neighbors(ent,hops=num_hops,avoid_high_degree_nodes=True):
            nbr_entities.add(nbr)
    return nbr_entities
def get_tuples_about_entities(candidate_entities):
    """
    得到和候选实体相邻的t,r，就是能直接组成三元s,r,t
    :param candidate_entities:
    :return:
    """
    tuples = set([])
    for s in candidate_entities:
        if s in knowledge_base.get_high_degree_entities():
            continue
        for t in knowledge_base.get_adjacent_entities(s):
            r =knowledge_base.get_relation(s,t)
            tuples.add((s,r,t))
    return tuples
def main(args):
    with open(args.input_examples,'r') as input_examples_file:
        with open(args.output_examples,'w',newline='') as output_file:
            reader = csv.DictReader(input_examples_file,delimiter='\t',fieldnames=['question','answer'])
            writer =csv.DictWriter(output_file,delimiter='\t',
                                   fieldnames=['question','qn_entities','ans_entities',
                                               'sources','relations','targets'])
            for id,row in enumerate(reader):
                answer = row['answer']
                ans_entities= answer.split("|")
                ans_entities=remove_invalid_ans_entities(ans_entities)

                question =row['question']
                qn_entities = question_parser.get_question_entities(question)
                #qn_entities=remove_invalid_ans_entities(qn_entities)先不进行此步骤
                qn_entities=remove_high_degree_qn_entities(qn_entities)
                #得到问题实体中，距离是Hops_From_Qn_entities的实体
                nbr_qn_entities =get_neighboring_entities(qn_entities,num_hops=Hops_From_Qn_Entities)

                #relevant_entities 表示从doc文档中查找相关的实体
                relevant_entities=search_index.search_doc(question,limit=Max_Relevant_Entities)

                #聚合从问题中提取出来的候选实体
                candidate_entities = qn_entities.union(nbr_qn_entities,relevant_entities)
                #如果候选实体过大，进行随机缩减
                if len(candidate_entities) > Max_Candidate_Entities:
                    candidate_entities =set(random.sample(candidate_entities,Max_Candidate_Entities))

                tuples=get_tuples_about_entities(candidate_entities)
                #如果从候选实体中得到的三元组过多，进行随机缩减
                if len(tuples)>Max_Candidate_Tuples:
                    tuples =set(random.sample(tuples,Max_Candidate_Tuples))

                #这里的source,relations,targets 就是从问题中进行实体提取出来之后，相关的的三元组
                #s,r,t的list是对应的 如(s[0],r[0],t[0])能构成合法的三元组
                sources = extract_dimension_from_tuples_as_list(tuples,0)
                relations= extract_dimension_from_tuples_as_list(tuples,1)
                targets=extract_dimension_from_tuples_as_list(tuples,2)

                output_raw={
                    'question':question,
                    'qn_entities':"|".join(qn_entities),
                    'ans_entities':"|".join(ans_entities),
                    'sources':"|".join(sources),
                    'relations':"|".join(relations),
                    'targets':"|".join(targets)
                }
                writer.writerow(output_raw)
                print("ok----",id)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_examples',default=absPath+"clean_wiki_qa_test.txt")
    parser.add_argument('--output_examples',default=absPath+"wiki_kv_qa_test.txt")
    args =parser.parse_args()
    input_graph = absPath+"clean_wiki_kb.txt"
    input_doc=absPath+"clean_wiki_doc.txt"
    stop_words=absPath+"wiki_stop_list.txt"
    #全局变量
    knowledge_base = KnowledgeGraph(input_graph,unidirectional=False)
    search_index = SearchEng(input_doc,stop_words)
    stop_vocab = read_file_as_dict(stop_words)
    question_parser = QuestionParser(knowledge_base.get_entities(),stop_list=stop_vocab)
    main(args)
