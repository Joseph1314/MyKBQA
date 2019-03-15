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
    def remove_substring(self,q_entities):
        """
        去除掉子串包含的情况 ，例如 like ,like you, -> like
        这种情况，应用到中文句子的时候，可能会更复杂一点
        :param q_entities:
        :return:
        """
        if len(q_entities)>1:
            q_ent_clean = set([])
            for ent1 in q_entities:
                flag = False #判断是否存在包含关系的标志
                for ent2 in q_entities:
                    if ent1 == ent2:
                        continue
                    if ent1 in ent2:
                        flag = True
                        break
                if not flag:#没有包含关系
                    q_ent_clean.add(ent1)
            q_entities = q_ent_clean
        return  list(q_entities)
    def get_unique_set(self,s1,s2):
        """
        得到两个集合不相交的部分，即各自与众不同的部分 unique
        :param s1:
        :param s2:
        :return:
        """
        intersection = s1.intersection(s2)
        for e in intersection:
            s1.remove(e)
            s2.remove(e)
        return s1,s2
    def remove_stop_word_and_score(self,s1,s2):
        """
        用于对剔除交集之后的集合s1,s2可能还会存在stop word
        对此进行一个剔除stop word 并且根据stop word的频率进行一个打分，最后一个选择
        :param s1:
        :param s2:
        :return:
        """
        score1,score2 = 0,0
        for w in list(s1):
            if w in self.stop_list:
                score1 = score1 +self.stop_list[w]
                s1.remove(w)
        for w in list(s2):
            if w in self.stop_list:
                score2 = score2 +self.stop_list[w]
                s2.remove(w)
        return s1,score1,s2,score2
    def remove_fake_entities(self,q_entities,question):
        """
        移除 重叠的fake 实体
        :param q_entities:
        :param question:
        :return:
        """
        if len(q_entities) > 1:
            q_ent_clean =set([])
            for ent1 in q_entities:
                for ent2 in q_entities:
                    if ent1 == ent2:
                        continue
                    #具体讲实体的单词细分
                    s1,s2 = set(ent1.split(" ")),set(ent2.split(" "))
                    pos1,pos2 = question.find(ent1),question.find(ent2)
                    intersection = s1.intersection(s2)
                    if len(intersection) == 0:
                        q_ent_clean.add(ent1)
                    if pos1 < pos2 and pos1 + len(ent1)>pos2:
                        #两个实体的选取的时候有重叠，造成了歧义，正常情况下，正确的语义划分只能有一种情况
                        #但是不确定是哪一个，所以进行一些选择

                        #step 1 既然有重叠，那么就要比较一下各自不重叠的部分
                        s1,s2 = self.get_unique_set(s1,s2)
                        s1,score1,s2,score2 = self.remove_stop_word_and_score(s1,s2)

                        #step 2 如果不重叠部分除掉stop_word仍然有word ,就保留实体
                        if len(s1)>0:
                            q_ent_clean.add(ent1)
                        if len(s2) >0:
                            q_ent_clean.add(ent2)

                        #step 3 如果当前没有找到任何一个实体，则至少需要添加一个
                        #将实体中包含stop word的出现频率最少的实体作为选择
                        if len(q_ent_clean) ==0:
                            if score1 <score2:
                                q_ent_clean.add(ent1)
                            else:
                                q_ent_clean.add(ent2)
            q_entities = q_ent_clean
        return list(q_entities)


    def get_question_entities(self,question):
        """
        根据输入的句子，抽取相关的实体
        :param question:
        :return:
        """
        q_entities = []
        q_words = question.split(" ")

        for length in tqdm(range(1,len(q_entities)+1)):
            i=0
            while i+length <= len(q_words):
                gen_ent = q_words[i:i+length]#生成的实体
                gen_ent = " ".join(gen_ent)
                if gen_ent in self.valid_entity_set:
                    q_entities.append(gen_ent)
                i = i+1

        """
        移除stop word ,子串，fake 实体  ，要知道valid实体中是包含stop word中的实体的，
        并且，通过暴力穷举生成的实体，一般认为实体的词汇量越多，代表更准确的实体
        例如 "姚明的女儿出生日期是哪天" 通过穷举识别出来['姚明'],['姚明的女儿'] 都是valid_entity_set中的实体
        但是显然['姚明的女儿']更加精确
        """
        q_entities = self.remove_all_stop_word_left_one(q_entities)#先去掉stop word
        q_entities = self.remove_substring(q_entities)#移除子串
        q_entities = self.remove_fake_entities(q_entities,question)
        return set(q_entities)
