

from torch.utils.data import DataLoader
import torch

from data_utils import *
import random
from tqdm import tqdm
# import pageRank
from collections import Counter

import numpy as np
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
from bert_score import score,BERTScorer

def turn2tensor_move2GPU(args, data):
    result = []
    
    for d in data:
        result.append({k:torch.tensor(v).to(args.device) for k,v in d.items()})
    return result

import torch.nn.functional as Func

class Similarity(object):

    def __init__(self,bert_score_ck,idf_sents, device='cuda', args=None, model=None, tokenizer=None):

        self.model = model
        self.config = args
        self.tokenizer = tokenizer
        self.bert_score_ck = bert_score_ck
        self.bert_scorer = BERTScorer(lang='zh', idf_sents = idf_sents, rescale_with_baseline=False, device=device,model_type=self.bert_score_ck,num_layers=12,idf=False)

    def data_prepare(self, passage_a: list, passage_b: list):
        passage_a_tokenized = self.tokenizer(passage_a,max_seq_length=512)
        passage_b_tokenized = self.tokenizer(passage_b,max_seq_length=512)

        passage_a_tensor = turn2tensor_move2GPU(passage_a_tokenized)
        passage_b_tensor = turn2tensor_move2GPU(passage_b_tokenized)

        passage_a_dataloader = DataLoader(passage_a_tensor, shuffle=False, batch_size=min(64, len(passage_a)))
        passage_b_dataloader = DataLoader(passage_b_tensor, shuffle=False, batch_size=min(64, len(passage_b)))

        return passage_a_dataloader, passage_b_dataloader

    
    def scale(self, F, temp=0.1,scale_way='out'):
        X = F / temp
        if scale_way == 'in':
            X = Func.softmax(X, dim = 0)
        else:
            X = Func.softmax(X, dim = 1)
        return X
    def txt_pair_score(self, txt_a,txt_b):
        hyps = [txt_a]
        refs = [txt_b]
        P, R, F = self.bert_scorer.score(hyps, refs, batch_size=64)
        return P,R,F
    def Bert_score(self, passage_a, passage_b):
        hyps = []
        for sent in passage_a:
            hyps+=[sent] * len(passage_a)
        refs = passage_b * len(passage_b)

        P, R, F = self.bert_scorer.score(hyps, refs, batch_size=96)

        P = P.reshape(len(passage_a), len(passage_b))
        R = R.reshape(len(passage_a), len(passage_b))
        F = F.reshape(len(passage_a), len(passage_b))
        P = P - torch.diag_embed(P.diag())
        P[: ,0] = 0
        # P[0,] = 1e-3
        R = R - torch.diag_embed(R.diag())
        R[:,0] = 0
        # R[0,] = 1e-3
        F = F - torch.diag_embed(F.diag())
        F[:,0] = 0
        # F[0,] = 1e-2
        # print(F)
        # F = F * 20
        
        return P,R,F


def search(graph, start, length = 10):
    path = [0]

    now = start
    path.append(now)
    
    while len(path) < length:
        # print(graph[now].shape)
        next_ind = torch.multinomial(graph[now], 1)
        # print(next_ind)
        now = next_ind[0]
        if now not in path:
            path.append(now)
        # else:
        #     print(path)
        # path.append(now)
    
    return path

def parse_directed(data):
    DG = nx.DiGraph()

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            node_a = i
            node_b = j

            DG.add_edge(node_a, node_b)
            DG.add_path([node_b, node_a])

        return DG


def A(data):
    try:
        for d in tqdm(data):
            d = [line.strip() for line in d if len(line.strip('。')) > 0]
            if len(d) <= 10:
                continue
            _,_,F = sim_cal.Bert_score(d, d, 'out')
            # print(F)
            
            ins = F.sum(dim = 0)
            # print(F)
            scores, ids  = ins.topk(ins.shape[0])
            path =search(F, ids[0]) 
            # print(path)
            # nodes+=[p if isinstance(p, int) else p.item()for p in path]
            # print(path)
            # print([d[p] for p in path])
            result.append([d[p] for p in path])
            print([d[p] for p in path])

            path =search(F, ids[-2]) 
            # print(path)
            # nodes+=[p if isinstance(p, int) else p.item()for p in path]
            # print(path)
            # print([d[p] for p in path])
            result.append([d[p] for p in path])
            print([d[p] for p in path])
    except:
        write_json(result, '../data/doc_graph_search_x.json')

    pass
def B(data):

    for d in tqdm(data):
        d = [line.strip() for line in d if len(line.strip('。')) > 0]
        if len(d) <= 10:
            continue
        _,_,F = sim_cal.Bert_score(d, d, 'out')
    #     # print(F)
        
        ins = F.sum(dim = 0)
    #     # print(F)
        scores, ids  = ins.topk(ins.shape[0])
        # for i in range(F.shape[0]):
        #     for j in range(F.shape[1]):
        #         _, id_X  = F[i,].topk(1)
        #         _, id_Y  = F[:,j].topk(1)
        #         if id_X == j and id_Y == i:
        #             print((i,j))


        print(d)
        # print([d[ind] for ind in ids])
        print(ids)
        print("\n入度最高的节点，即最容易被访问的节点：\n",[d[int(ind.item()) ] for ind in ids])
        # # print('*' * 50)
        # print(d)
        # print(F)
        nodes = []
        # path =search(F, 0) 
        # print(path)
        # nodes+=[p if isinstance(p, int) else p.item()for p in path]
        # print(path)
        # print([d[p] for p in path])
        # result.append([d[p] for p in path])
        
    #     for k in tqdm(range(5)):
    #         path =search(F, ids[0]) 
    #         print(path)
    #         nodes+=[p if isinstance(p, int) else p.item()for p in path]
    # pass
