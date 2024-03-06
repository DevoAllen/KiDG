
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from data_utils import *
from similarity import *
from tqdm import tqdm
import operator
import functools
import torch.nn.functional as Func
from argparse import ArgumentParser

def get_txt_pair_bertscore(text_a, text_b, bert_scorer):
    
    P,R,F = bert_scorer.txt_pair_score(text_a, text_b)

    # print('*'*10)
    # print(text_a, text_b, F)

    return P,R,F



def scale(links):
    #将句子间的相似度转化为概率
    links = Func.softmax(links, dim = 1)
    # 将文档内全连接图处理,按照出度scale
    # 文档间连接因为每次用需要重新处理，因此不做概率转换
    return links

def get_key_nodes(links):
    ins = links.sum(dim = 0)
    # print(F)
    scores, ids  = ins.topk(max(ins.shape[0]//5, min(2, ins.shape[0])))
    return ids.tolist()

import logging

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)



from torch.multiprocessing import Pool, Process, set_start_method, Manager
if __name__=='__main__':
    set_start_method('spawn',force=True)

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--intra_path', type=str, default='../data/IntraDocumentRelations/docs_F.pkl')
    arg_parser.add_argument('--inter_path', type=str, default='../data/InterDocumentRelations/inter_relations.json')  
    arg_parser.add_argument('--simcse', type=str, help="path_for_storing_bert_score_results", default="../supply/sentEmbed/")
    arg_parser.add_argument('--output_path', type=str, default='../data/ResultGraph')
    
    


    args = arg_parser.parse_args()


    

    file_bert_score_path = args.intra_path

    doc_relation_json_path =  args.inter_path
    doc_relations = read_json_origin(doc_relation_json_path)

    logger.info(f"finished reading inter-documents relations")     


    logger.info(f"finished reading documents")     
    data_bert_score = read_pickle(file_bert_score_path)
    data = [ds[0] for ds in data_bert_score]

    
    logger.info(f"finished reading intra-document graph")     

    result = []
    
    unit_len = len(doc_relations) // args.splits
    
    cuda_n = torch.cuda.device_count()
    
    
    now = 0
    from tqdm import tqdm
    num = 0

    doc_relations_items = [(k,v) for k,v in doc_relations.items()]
    logger.info(f'There are {len(doc_relations_items)} items.')

    
    result = []
    for i in range(len(data)):
        result.append({})
    sim_cal = Similarity(bert_score_ck = args.simcse,device="cuda", idf_sents = None )

    for major, minors in tqdm(doc_relations_items):
        major = int(major)
        maj_data = data_bert_score[major][0]
        maj_score = data_bert_score[major][1]

        
        maj_score = scale(maj_score)
        maj_key_nodes = get_key_nodes(maj_score)

        for min_id, minor_list in minors.items():
            min_id = int(min_id)
            if isinstance(minor_list, dict):
                minor_list = [minor_list]
            for minor in minor_list:
                
                if minor['type']=='hyper_link':

                    text_a_id = minor['info'][0]


                    if len(maj_data) < text_a_id:
                        continue
                    text_a = maj_data[text_a_id]
                    if len(data[min_id]) > 1:
                        text_b = data[min_id][1]
                    else:
                        text_b = data[min_id][0]

                    p,r,f = get_txt_pair_bertscore(text_a, text_b, sim_cal)
                    if text_a_id not in result[major].keys():
                        result[major][text_a_id] = {f'{min_id}-{1}':f[0].item()}
                    else:
                        result[major][text_a_id][f'{min_id}-{1}'] = f[0].item()
                else:
                    min_data = data_bert_score[min_id][0]
                    min_score = data_bert_score[min_id][1]
                    min_score = scale(min_score)
                    min_key_nodes = get_key_nodes(min_score)
                    for maj_key_node in maj_key_nodes:
                        text_a = maj_data[maj_key_node]
                        scores = [get_txt_pair_bertscore(text_a, min_data[text_b_id], sim_cal) for text_b_id in min_key_nodes]
                        # print(scores)

                        for s, min_sent_id in zip(scores,min_key_nodes):
                            if maj_key_node not in result[major].keys():
                                result[major][maj_key_node] = {f'{min_id}-{min_sent_id}':s[2].item()}
                            else:
                                result[major][maj_key_node][f'{min_id}-{min_sent_id}'] = s[2].item()
                
    dump_pickle(result, f'{args.output_path}/kdconv_cross.pkl')
    logger.info(f"Results are stored in {args.output_path}/kdconv_cross.pkl !")
