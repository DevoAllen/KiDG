import numpy as np
import torch
import torch.nn.functional as Func
from data_utils import *
import random
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class Link(object):
    def __init__(self, major, minors):
        self.texts = in_doc_links[major][0]
        self.in_links = in_doc_links[major][1]
        # print(self.in_links)
        self.cross_links = cross_doc_links[major]
        self.scale(temp = 0.2)

    def scale(self, temp = 1):
        #将句子间的相似度转化为概率
        # print(self.in_links.shape)
        self.in_links = Func.softmax(self.in_links / temp, dim = 1)
        # 将文档内全连接图处理,按照出度scale
        # 文档间连接因为每次用需要重新处理，因此不做概率转换
        

    def get_key_nodes(self):
        ins = self.in_links.sum(dim = 0)
        # print(F)
        scores, ids  = ins.topk(max(ins.shape[0]//5, 1))
        #TODO： 这里的这个8最好跟整个数据库的平均句子数量有关，否则不会太好
        return ids.tolist()

class Node(object):
    #一个Node是一次主文档处理的单位
    def __init__(self, major: int, minors: list):
        self.major = major
        self.minors = minors
        self.links = Link(major, minors)
        self.key_nodes = self.links.get_key_nodes()
        self.texts = self.links.texts


    

def prepare_starts():
    nodes = []
    for k,v in doc_relations.items():
        nodes.append(Node(int(k), v))
    return nodes

        


def judge_add(now, outways,out_sents, Where):
    if now in outways.keys():
        out_sents=torch.cat((out_sents, torch.tensor([v for v in outways[now].values()]))) 
        Where = Where +[k for k in outways[now].keys()]
        return True, out_sents, Where
    else:
        return False, out_sents, Where
def search(major_node, start, length = 10):
    #TODO：一次主文档遍历，只考虑主文档内遍历多少句子
    # print(major_node.minors)

    length = major_node.links.in_links.shape[0]
    # print(length)
    b = int( length)
    times = 0

    Where = ['No']
    now = start
    path = []
    
    outways = major_node.links.cross_links
    out_sents = torch.tensor([])
    # print(outways)
    anchor_or_no,out_sents,Where =  judge_add(now, outways, out_sents, Where)
    if anchor_or_no:
        times+=1
    
    
    for i in range(1,30):
        next_ind = torch.multinomial(major_node.links.in_links[now], 1)
        
        now = next_ind[0]
        if f'{major_node.major}-{now}' not in path:
            path.append(f'{major_node.major}-{now}')
            # print(now, outways.keys())
            anchor_or_no,out_sents,Where =  judge_add(now, outways, out_sents, Where)
            if anchor_or_no:
                times+=1

        
        temp = i/b
        # 
        W = (1/max(1, times)) * out_sents
        # print(W)
        if len(W)!=0:
            P = Func.softmax( torch.cat( ( torch.tensor([min( max(W)+1 , 1.)]),W)) / temp )
        else:
            P=torch.tensor([1.])

        next_act = torch.multinomial(P, 1)[0]

        if next_act!=0:
            return path, Where[next_act]
    return path, Where[0]

def process(start_node, max_jump=5, max_sents=25):
    
    jump = 0
    
   
    start = random.choice([0]+start_node.key_nodes)
    path = [f'{start_node.major}-{start}']


    major_node = start_node
    now = start
    while jump < max_jump and len(path) < max_sents:
        tmp_path,act = search(major_node, now)
        for tp in tmp_path:
            if tp not in path:
                path.append(tp)
        if act == 'No':
            # if len(path) < max_sents:
            #     # major_node = nodes[doc_id2node[doc_id]]
            #     now = int(path[-1].split('-')[-1])
            # elif len(path) >= max_sents:
            #     return path
            return path
        else:
            jump+=1
            doc_id = int(act.split('-')[0])
            sent_id = int(act.split('-')[1])
            if doc_id not in doc_id2node:
                return path
            major_node = nodes[doc_id2node[doc_id]]
            now = sent_id
            if f'{major_node.major}-{now}' not in path:
                path.append(f'{major_node.major}-{now}')
    return path

def get_sentence_seq(path):
    sentences = []

    now_doc = -1
    now_doc = int(path[0].split('-')[0])
    now_title = nodes[doc_id2node[now_doc]].texts[0]
    s = random.choice(starts)

    sentences.append(s + now_title+'吗？')

    passed_id = [now_doc]
    for act in path:
        doc_id = int(act.split('-')[0])
        sent_id = int(act.split('-')[1])
        if doc_id != now_doc:
            
            from_doc_id = now_doc
            from_doc_title = now_title
            
            if str(doc_id) not in doc_relations[str(now_doc)].keys():
                    for p_id in passed_id:
                        if str(doc_id) in doc_relations[str(p_id)]:
                            from_doc_id = p_id
                            from_doc_title =  nodes[doc_id2node[p_id]].texts[0]
                            break
            

            relations = doc_relations[str(from_doc_id)][str(doc_id)]

            
            relation = relations[0]['type']
            
            info = relations[0]['info']
            for rel in relations:
                if rel['type'] == 'rela':
                    relation = rel['type']
                    info = rel['info']
                    info = [str(nf) for nf in info]
            if relation=='sim' or relation=='hyper_link' or (relation=='rela' and len(''.join(info))==0 ):
                nex_title = nodes[doc_id2node[doc_id]].texts[0]

                question = random.choice(rela_questions)
                answer = random.choice(rela_answers)

                sentences.append('---==='+question)
                sentences.append(f'---==={answer[0]}{nex_title}{answer[1]}')
            else:
                info = info[0]
                nex_title = nodes[doc_id2node[doc_id]].texts[0]

                inds = list(range(len(kg_questioins)))
                ind = random.choice(inds)
                question = kg_questioins[ind]
                answer = kg_answers[ind]
                if ind==0:
                    sentences.append('---===' + f'{question[0]}{from_doc_title}{question[1]}{info}{question[2]}')
                    sentences.append(f'---==={answer[0]}{nex_title}{answer[1]}')
                else:
                    sentences.append('---===' + f'{from_doc_title}{question[0]}{info}{question[1]}{nex_title}{question[2]}')
                    sentences.append(f'---==={answer[0]}')
            now_doc = doc_id
            now_title = nex_title
            passed_id.append(now_doc)

        sentences.append(nodes[doc_id2node[doc_id]].texts[sent_id])
    return sentences

def high_process(node):

    have = set()

    sent_num = len(node.texts)
    k = 0
    temp_results = []
    while k < 5 or len(list(have)) < sent_num:
        path = process(node)
        temp_results.append(get_sentence_seq(path))

        for p in path:
            if p.startswith(f'{node.major}'):
                # print(p)
                have.add(p)
        k+=1
    return temp_results
    
# from torch.multiprocessing import Pool
from argparse import ArgumentParser
from torch.multiprocessing import Pool

from torch.utils.data import DataLoader





if __name__=='__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--intra_path', type=str, default='../data/IntraDocumentRelations/kdconv_intra_relations_F.pkl')
    arg_parser.add_argument('--inter_path', type=str, default='../data/InterDocumentRelations/kdconv_inter_relations.json')  
    arg_parser.add_argument('--graph_path', type=str, default='../data/ResultGraph.kdconv_cross.pkl') 
    arg_parser.add_argument('--output_path', type=str, default='../output')
    

    args = arg_parser.parse_args()
    from tqdm import tqdm
    doc_id2node = {}

    in_doc_links = [[],[]]

    cross_doc_links = [{},{}]

    doc_relations = [[]]
    
    doc_bert_score_pkl_path = args.intra_path
    doc_relation_json_path = args.inter_path
    doc_cross_link_pkl_path = args.graph_path


    in_doc_links = read_pickle(doc_bert_score_pkl_path)

    logger.info(f"Read Intra-relations")

    cross_doc_links = read_pickle(doc_cross_link_pkl_path)
    logger.info(f"Read Inter-relations")

    doc_relations = read_json_origin(doc_relation_json_path)
    logger.info(f"Read specific relations between docs.")


    nodes = prepare_starts()
    for ind, node in enumerate(nodes):
        doc_id2node[node.major] = ind

    rela_questions = ['我们来聊聊其他的东西？','我对你刚才说的不太感兴趣，聊点别的？','你还知道些什么？']
    rela_answers = [["我了解一些关于",'的事。'],['再聊聊','？']]

    kg_questioins = [["你知道",'的','吗？'],['的','是','吧。']]
    kg_answers = [["了解过，是",'吧。'],['是的。']]
    starts = ['知道','有了解']

    result = []

    threads = 64

    ids = list(range(len(nodes)))


    unit_len = len(ids) // args.splits

    for ind in tqdm(ids[:10]):
        result+=high_process(nodes[ind])  
        
    with open(f'{args.output_path}/KiDial.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
