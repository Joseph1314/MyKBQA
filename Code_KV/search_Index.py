"""

    用来实现一个
    搜索文档doc类型的 搜索引擎，使用whoosh 的库搭建一个搜索引擎
    可以轻松实现一个搜索的功能
"""
import codecs
import unicodedata
from word_process_method import *
from whoosh import qparser
from whoosh import scoring
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.filedb.filestore import RamStorage
class SearchEng(object):
    """
    一个搜索index的基于whoosh搭建的搜索引擎的class
    """
    def __init__(self,doc_file,stop_words_path=None):
        st = RamStorage()#存储对象保存在内存中
        st.create()
        schema = Schema(entity_name=TEXT(stored=True),
                        fieldname=TEXT(stored=True),content=TEXT())
        self.ix = st.create_index(schema)
        writer = self.ix.writer()#写入文件
        self.remove_stop_word_while_indexing = False
        if stop_words_path:
            self.remove_stop_word_while_indexing = True
            self.stop_list = read_file_as_dict(stop_words_path)
        with codecs.open(doc_file,'r',"utf-8") as doc_file:
            for line in doc_file:
                line = clean_line(line)
                ent,field,content = line.split("|")
                writer.add_document(entity_name=ent,
                                    fieldname=field,
                                    content=self.remove_stop_word_from_text(content))
        writer.commit()
    def remove_stop_word_from_text(self,content):
        words = content.split(" ")
        words_clean = []
        for w in words:
            if self.remove_stop_word_while_indexing and w not in self.stop_list:
                words_clean.append(w)
        return " ".join(words_clean) if len(words_clean) >0 else content
    def search_doc(self,question,limit=20):
        """
        搜索包含问题中的词的三元组doc，返回每个三元组的实体名称
        :param question:
        :param limit: 限制搜索到的结果
        :return:
        """
        results = set([])
        question = self.remove_stop_word_from_text(question)# " ".join 连接的词列表
        with self.ix.searcher() as searcher:
            query = QueryParser("content",self.ix.schema,group=qparser.OrGroup).parse(question)
            res = searcher.search(query,limit=limit)
            for r in res:
                results.add(r['entity_name'])
        results = [unicodedata.normalize('NFKD',res).encode('utf-8','ignore').decode('utf-8') for res in results]
        return results
if __name__ == "__main__":
    doc_path="../data/movieqa/ac_doc.txt"
    stop_path="../data/movieqa/stop_list.txt"
    search = SearchEng(doc_path,stop_path)
    print(search.search_doc("ginger rogers and"))