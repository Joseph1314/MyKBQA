"""
    对输入的问题进行解析,最终能获取到问题中的实体
"""
import argparse
import csv
from tqdm import tqdm
class QuestionParser(object):#新式类
    """
        解析问题
    """
    def __init__(self,valid_entity_set,stop_list):
        self.valid_entity_set = valid_entity_set
        self.stop_list = stop_list
    def remove_all_stop_word_left_one(self,q_entities):
        """
        对于输入问题进行移除stop word 但是得至少保留一个单词，选择相对来说出现频率最小的word
        :param q_entities:
        :return:
        """
        q_entities_clean = set([])
        for ent in q_entities:
            if ent not in self.stop_list:
                q_entities_clean.add(ent)

        #如果所有的都是stop word 选择一个出现频率最小的
        if len(q_entities_clean) == 0:
            min_frq_ent =q_entities[0]
            for ent in q_entities:
                if self.stop_list[ent] < self.stop_list[min_frq_ent]:
                    min_frq_ent = ent
            q_entities_clean.add(min_frq_ent)
        return q_entities_clean
