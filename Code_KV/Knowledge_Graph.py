import argparse
import csv
import networkx as nx
from word_process_method import *
High_Degree_Threshold = 50#高度节点的阈值
class KnowledgeGraph(object):
    def __init__(self,graph_path, unidirectional = True):
        """
            unidirectional =True 即无向图，需要把逆三元组也加入到KB中
            但是为了能表示入度和出度，仍然使用有向图格式
        """
        self.G = nx.DiGraph()
        with open(graph_path,'r',encoding='utf-8') as graph_file:#也即KB文件
            cnt =0
            nodes=0
            for line in graph_file:
                line = line.strip("\n")
                e1,relation,e2 = line.split("|")
                cnt = cnt+1
                flag =False
                self.G.add_edge(e1,e2,relation=relation)
                #if flag:
                #    print("->",self.G[e1][e2]['relation'])
                if not unidirectional:
                    #nodes=nodes+1
                    #print("kkkkk")
                    self.G.add_edge(e2,e1,relation=self.get_inverse_relation(relation))
                    #print("!->", self.G[e2][e1]['relation'])
        print("len:",cnt)
        self.high_degree_nodes = set([])
        indeg = self.G.in_degree()#返回的是元组的形式 two-tuples of (node, in-degree).
        for k,v in indeg:
            if v >High_Degree_Threshold:
                self.high_degree_nodes.add(k)
        self.all_entities = set(nx.nodes(self.G))#获取图中的所有顶点
    def get_inverse_relation(self,relation):
        return "!_"+relation
    def get_entities(self):
        return self.all_entities
    def get_all_paths(self,source,target,cut_off):
        """
        得到从source节点到target节点的
        1->用节点表示的路径信息
        2->用Edge(relation) 表示的路径信息
        :param source: 源节点
        :param target: 目标节点
        :param cut_off: 最大的路径长度
        :return:
        """
        if source == target:
            return [],[]
        paths_of_entities = []
        paths_of_relations = []
        for path in nx.all_simple_paths(self.G,source,target,cut_off):#[0,1,2] path 是一个list的形式
            paths_of_entities.append(path)
            relation_of_this_path=[]
            for i in range(0,len(path)-1):
                relation  = self.G[path[i]][path[i+1]]['relation']
                relation_of_this_path.append(relation)
            paths_of_relations.append(relation_of_this_path)
        return paths_of_entities,paths_of_relations
    def get_candidate_neighbors(self,start_node,hops=2,avoid_high_degree_nodes=True):
        """
        使用BFS方法，获得start_node 距离为hops的邻接节点
        并且可选避免使用高度节点
        :param start_node:
        :param hops:
        :param avoid_high_degree_nodes:
        :return:
        """
        result = set([])
        q = [start_node]
        visited = set([start_node])
        distance = {start_node:0}
        while len(q) > 0:
            u=q.pop(0)
            #需要判断u是否在G中
            if u in self.G.nodes():
                result.add(u)
                for nbr in self.G.neighbors(u):
                    if nbr in self.high_degree_nodes and avoid_high_degree_nodes:
                        continue
                    if nbr not in visited :
                        visited.add(nbr)
                        distance[nbr]=distance[u]+1
                        if distance[nbr] <= hops:
                            q.append(nbr)
        if start_node in result:
            result.remove(start_node)
        return result
    def get_adjacent_entities(self,node):
        return set(self.G.neighbors(node))
    def get_high_degree_entities(self):
        return self.high_degree_nodes
    def get_relation(self,source,target):
        return self.G[source][target]['relation']
if __name__ =="__main__":
    graph_path = "../data/movieqa/ac_kb.txt"
    kb = KnowledgeGraph(graph_path, unidirectional=False)
    print(len(kb.all_entities))
    entities_paths, relations_paths = kb.get_all_paths(source='moonraker', target='lewis gilbert', cut_off=3)
    print(kb.get_candidate_neighbors("moonraker"))
    print(len(kb.get_candidate_neighbors("moonraker", hops=2)))
    #kb.log_statistics()
    print(kb.get_candidate_neighbors("what", hops=1))
    for path in kb.get_all_paths('bruce lee', 'bruce lee', cut_off=2):
        print(path)


