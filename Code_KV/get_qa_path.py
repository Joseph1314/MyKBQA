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
    