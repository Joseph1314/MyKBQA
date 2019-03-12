"""
    将问题，答案，实体，关系等
    映射成 word 级别的字典，原子级别，
    每一种类型（word级别和实体关系级别的，都有一个索引id对应
    比如，"Joseph Tom" 作为一个实体，有一个id索引对应，"Joseph" 和 "Tom" 单个单词各自有一个索引对应
"""
from collections import defaultdict
import random
import csv
from word_process_method import *
import tqdm
word = set([]) #单词级别
entities = set([])
relations = set([])
