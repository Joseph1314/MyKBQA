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
        return self.qp.get_question_entities()
    def get_relevant_entities_from_index(self,question):
        if not USE_RELEVANT_ENTITIES:
            return set([])
        result = self.index.search_doc(question,limit=COUNT_RELEVANT_ENTITIES)
        print(len(result),result)
        return result
    def get_neighboring_entities(self,entities,num_hops=2):
        """
        得到entities的所有距离为num_hops的entity
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
        return self.kb.get_all_paths(qn_entity,candidate_entity)

    def get_all_paths_many(self,qn_entities,candidate_entities):
        all_paths_of_entities =[]#节点类型
        all_paths_of_relations =[]#边类型
        max_candidate_paths_from_sigle_pair = max(1,MAX_CANDIDATE_PATHS/len(candidate_entities))
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
        return all_paths_of_entities,all_paths_of_relations

def sample_paths(path_of_entities,path_of_relations,cut_off):
    """
    随机截取cut_off 长度的list 从原来的地方
    :param path_of_entities:
    :param path_of_relations:
    :param cut_off:
    :return:
    """
    sample_idx=set(random.sample(range(0,len(path_of_entities)),cut_off))
    sample_paths_of_entites =[]
    sample_paths_of_relations =[]
    for idx in sample_idx:
        sample_paths_of_entites.append(path_of_entities[idx])
        sample_paths_of_relations.append(path_of_relations[idx])
    return sample_paths_of_entites,sample_paths_of_relations
def main(args):
    ee=EntityExtractor(args.kb,args.doc,args.stop_list)
    with open(args.input_qa_pair,'r') as input_qa_file:
        with open(args.output_qa_examples,'w') as out_qa_file:
            """
                生成具有更加详细信息的数据集，具有路径，相关节点，邻接节点
            """
            reader = csv.DictReader(input_qa_file,delimiter='\t',fieldnames=['question','answer'])
            writer = csv.DictWriter(out_qa_file,delimiter='\t',
                                    fieldnames=['question','qn_entities','ans_entites','relevant_entities',
                                                'nbr_qn_entities','nbr_relevant_entities','candidate_entities',
                                                'path_of_entities','path_of_relations'])
            writer.writeheader()#Write a row with the field names (as specified in the constructor).
            max_count_candidate_entities = 0



