"""
    用来生成包含路径的数据集

"""
import argparse
import csv
import random
from word_process_method import *
from Knowledge_Graph import KnowledgeGraph
from search_Index import SearchEng
from parse_question import QuestionParser
from tqdm import tqdm

#跳数设置
HOPS_FROM_QN_ENTITY = 2
HOPS_FROM_RELEVANT_ENTITY = 1

#相关参数设置
MAX_CANDIDATE_ENTITIES = 128
MAX_CANDIDATE_PATHS = 1024
CLIP_CANDIDATE_PATHS_BETWEEN_SINGLE_PAIR = True

USE_RELEVANT_ENTITIES = True
USE_NBR_QN_ENTITIES = True #永远保持true,否则可能会没有可找到的路径
USE_NBR_RELEVANT_ENTITIES = True
#USE_NBR_ANSWER_ENTITIES

REMOVE_HIGH_DEGREE_ANSWER_ENTITIES = False #what was the release date of the film almighty thor
MAX_PATH_LENGTH = 3 #this includes the source and target entities, 3 translates to 1 intermediate node
COUNT_RELEVANT_ENTITIES = 20

class EntityExtractor(object):
    def __init__(self,input_graph,input_doc,stop_list):
        self.kb = KnowledgeGraph(input_graph,unidirectional=False)
        self.index = SearchEng(input_doc,stop_list)
        valid_entities_set = self.kb.get_entities()
        stop_vocab = read_file_as_dict(stop_list)
        self.qp = QuestionParser(valid_entities_set,stop_vocab)
    def get_question_entities(self,question):
        return self.qp.get_question_entities(question)
    def get_relevant_entities_from_index(self,question):
        """
        搜索doc文件中的（entitiy,field,content)三元组的entity，条件是至少包含一个问题中的word
        :param question:
        :return:
        """
        if not USE_RELEVANT_ENTITIES:
            return set([])
        result = self.index.search_doc(question,limit=COUNT_RELEVANT_ENTITIES)
        print((len(result),result))
        return result
    def get_neighboring_entities(self,entities,num_hops=2):
        """
        从KB图结构的三元组中，得到entities的所有距离为num_hops的entity
        :param entities:
        :param num_hops:
        :return:
        """
        nbr_entities = set([])
        for ent in entities:
            for nbr in self.kb.get_candidate_neighbors(ent,hops=num_hops):
                nbr_entities.add(nbr)
        return nbr_entities
    def get_neighboring_qn_entities(self,qn_entities):
        if not USE_NBR_QN_ENTITIES:
            return set([])
        return self.get_neighboring_entities(qn_entities,num_hops=HOPS_FROM_QN_ENTITY)
    def get_neighboring_relevant_entities(self,relevant_entites):
        if not USE_RELEVANT_ENTITIES:
            return set([])
        return self.get_neighboring_entities(relevant_entites,num_hops=HOPS_FROM_RELEVANT_ENTITY)
    def remove_high_degree_qn_entities(self,qn_entities):
        """
        对于一个问题来说，如果有多个实体，选择去除其中的高度节点
        如果每个节点都是高度节点，则随机选择一个
        :param qn_entities:
        :return:
        """
        if len(qn_entities) and REMOVE_HIGH_DEGREE_ANSWER_ENTITIES:
            qn_entities_clean = set([])
            for q in qn_entities:
                if q in self.kb.high_degree_nodes:
                    continue
                qn_entities_clean.add(q)
            if len(qn_entities_clean) >0:
                qn_entities=qn_entities_clean
            else :
                qn_entities = qn_entities_clean
        return qn_entities

    def get_paths_one2one(self,qn_entity,candidate_entity):
        return self.kb.get_all_paths(qn_entity,candidate_entity,cut_off=MAX_PATH_LENGTH)

    def get_all_paths_many(self,qn_entities,candidate_entities):
        all_paths_of_entities =[]#节点类型
        all_paths_of_relations =[]#边类型
        max_candidate_paths_from_sigle_pair = int(max(1,MAX_CANDIDATE_PATHS/len(candidate_entities)))
        for ans_ent in candidate_entities:
            ans_ent = clean_word(ans_ent)
            for qn_ent in qn_entities:
                if qn_ent in self.kb.get_entities() and ans_ent in self.kb.get_entities():
                    path_of_entities,path_of_relations = self.get_paths_one2one(qn_ent,ans_ent)
                    if len(path_of_entities) > max_candidate_paths_from_sigle_pair and \
                        CLIP_CANDIDATE_PATHS_BETWEEN_SINGLE_PAIR:
                        path_of_entities,path_of_relations = sample_paths(path_of_entities,path_of_relations,max_candidate_paths_from_sigle_pair)
                    all_paths_of_entities.extend(path_of_entities)
                    all_paths_of_relations.extend(path_of_relations)
        print(all_paths_of_entities)
        return all_paths_of_entities,all_paths_of_relations
def change2str(str):
    """
    将bytes 转化为str
    :param str:
    :return:
    """
    if type(str) == bytes:
        str = str.decode('utf-8')
        print("decode:",str)
    return str
def sample_paths(path_of_entities,path_of_relations,cut_off):
    """
    随机截取cut_off 长度的list 从原来的地方
    :param path_of_entities:
    :param path_of_relations:
    :param cut_off:
    :return:
    """
    sample_idx=set(random.sample(list(range(0,len(path_of_entities))),cut_off))
    sample_paths_of_entites =[]
    sample_paths_of_relations =[]
    for idx in sample_idx:
        sample_paths_of_entites.append(path_of_entities[idx])
        sample_paths_of_relations.append(path_of_relations[idx])
    return sample_paths_of_entites,sample_paths_of_relations
def main(args):
    ee=EntityExtractor(args.kb,args.doc,args.stop_list)
    with open(args.input_qa_pair,'r') as input_qa_file:
        with open(args.output_qa_examples,'w',newline='') as out_qa_file:
            """
                生成具有更加详细信息的数据集，具有路径，相关节点，邻接节点
            """
            reader = csv.DictReader(input_qa_file,delimiter='\t',fieldnames=['question','answer'])
            writer = csv.DictWriter(out_qa_file,delimiter='\t',
                                    fieldnames=['question','qn_entities','ans_entities','relevant_entities',
                                                'nbr_qn_entities','nbr_relevant_entities','candidate_entities',
                                                'path_of_entities','path_of_relations'])
            writer.writeheader()#Write a row with the field names (as specified in the constructor).

            max_count_candidate_entities = 0
            max_count_candidate_paths = 0
            for row in tqdm(reader):
                q_str= row['question'].strip("\n")
                if len(q_str) ==0:
                    continue
                #需要保证读取的不为空
                #print(("-------"+row['question']+"--------------"))
                qn_entities = ee.get_question_entities(row['question'])
                #print(("len:", len(qn_entities)))
                qn_entities = ee.remove_high_degree_qn_entities(qn_entities)
                ans_str = row['answer']
                ans_entities = ans_str.split("|")#答案的集合

                # 从doc文件中搜索至少包含一个question 中的word的三元组实体
                relevant_entities = ee.get_relevant_entities_from_index(row['question'])

                #从 kb文件中，BFS搜索跳数是2的临近实体
                nbr_qn_entities = ee.get_neighboring_entities(qn_entities)

                #将从doc文件中搜索到的相关实体，再次放到kb图结构中，搜索相应跳数的临近节点
                nbr_relevant_entities = ee.get_neighboring_relevant_entities(relevant_entities)

                #通过以上的查找，可以生成比较全面的答案候选实体，将其进行去并集,这里并不包含答案实体
                candidate_entites = qn_entities.union(relevant_entities,nbr_qn_entities,nbr_relevant_entities)

                if args.mode == "train":
                    candidate_entites = candidate_entites.union(ans_entities)
                #确定候选实体的维度
                max_count_candidate_entities = max(max_count_candidate_entities,len(candidate_entites))

                #减少候选实体的维度，随机选择其中的子集
                if len(candidate_entites) > MAX_CANDIDATE_ENTITIES:
                    candidate_entities = set(random.sample(candidate_entites,MAX_CANDIDATE_ENTITIES))
                all_paths_of_entities , all_paths_of_relations = ee.get_all_paths_many(qn_entities,candidate_entites)
                max_count_candidate_paths = max(max_count_candidate_paths,len(all_paths_of_entities))

                #减少候选路径的维度，通过随机选择他的子集
                if len(all_paths_of_entities) > MAX_CANDIDATE_PATHS:
                    all_paths_of_entities ,all_paths_of_relations = sample_paths(all_paths_of_entities,all_paths_of_relations,MAX_CANDIDATE_PATHS)
                output_row ={
                    'question':row['question'],
                    'qn_entities': "|".join(qn_entities),
                    'ans_entities': "|".join(ans_entities),
                    'relevant_entities': "|".join(change2str(relevant_entities)),
                    'nbr_qn_entities': "|".join(change2str(nbr_qn_entities)),
                    'nbr_relevant_entities': "|".join(change2str(nbr_relevant_entities)),
                    'candidate_entities': "|".join(change2str(candidate_entities)),
                    'path_of_entities': get_str_of_nested_seq(all_paths_of_entities),
                    'path_of_relations': get_str_of_nested_seq(all_paths_of_relations),
                }
                writer.writerow(output_row)
        print(("max count candidate entities",max_count_candidate_entities))
        print(("max count candidate entities",max_count_candidate_paths))
if __name__ =="__main__":

    parser =argparse.ArgumentParser()
    parser.add_argument('--mode',help='train or test mode ',default='train')
    parser.add_argument('--kb',help='kb form file',
                        default="../data/movieqa/ac_kb.txt")
    parser.add_argument('--doc',help='doc form file',
                        default="../data/movieqa/ac_doc.txt")
    parser.add_argument('--stop_list',help='stop word list',
                        default="../data/movieqa/stop_list.txt")
    parser.add_argument('--input_qa_pair',help='input question-answer data set',
                        default="../data/movieqa/clean_qa_full_train.txt")
    parser.add_argument('--output_qa_examples',help='generate qa data set with some detail path information',
                        default="../data/movieqa/train_qa_with_detail_information.txt")
    args =parser.parse_args()
    main(args)
