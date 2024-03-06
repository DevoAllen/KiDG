'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-07 22:43:42
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-03-06 15:36:58
FilePath: /wangrui/A2024/KDial-ACL2023/KiDG/code/intraRelations.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from data_utils import *
from similarity import *
from tqdm import tqdm
from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def doc2score(file_path, out_paths):
    data = read_json(file_path)

    sim_cal = Similarity(bert_score_ck = BERT_CKP,idf_sents = None )

    result_P = []
    result_R = []
    result_F = []
    with tqdm(data) as t:
        t.set_description('BertScore')
        for d in t:
            try:
                P,R,F = sim_cal.Bert_score(d, d)
                result_P.append((d, P))
                result_R.append((d, R))
                result_F.append((d, F))
                results = (result_P,result_R,result_F)
            except:
                continue
    
    for result, path in zip(results, out_paths):
        dump_pickle(result, path)


if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--doc_path', type=str, default="../data/Documents/docs.json",help="path_to_documents")
    arg_parser.add_argument('--output_path', type=str, default="../data/InterDocumentRelations", help="")
    arg_parser.add_argument('--simcse', type=str, help="path_for_storing_bert_score_results", default="../supply/sentEmbed/")
    args = arg_parser.parse_args()
    
    
    BERT_CKP = args.simcse

    
    out_path = [os.path.join(f'{args.output_path}', args.doc_path.replace('.json',f'_{x}.pkl')) for x in ['P','R','F'] ]
    
    doc2score(args.doc_path, out_path)  
    logger.info(f"Results are stored in {out_path}")


