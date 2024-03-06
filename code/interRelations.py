
from data_utils import *
from tqdm import tqdm
import os
import gensim
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from gensim.test.utils import datapath, get_tmpfile
from gensim.models  import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def findHyperlink(docs):
    # process a list of sentence-list
    
    
    doc_kws = []
    for doc in docs:
        temp = {}
        temp['doc'] = doc
        temp['kw']=[]
        for sent in doc[1:]:
            entry = sent.split("是")[-1].strip()
            if entry!= doc[0] and entry in doc_names:
                temp['kw'].append(entry)
        doc_kws.append(temp)
        
    title_keywords = { t["doc"][0]: list(set(t['kw'])) for t in doc_kws if len(list(set(t['kw']))) > 0}
    
    final_relations={}
    logger.info("Build the hyper-link relation, please wait")
    for title, keywords in tqdm(title_keywords.items()):
        main_doc_id = str(doc2id[title])
        
        assert title == id2doc[int(main_doc_id)]
        doc = docs[int(main_doc_id)]
        # print(doc)
        final_relations[main_doc_id]={}
        
        keywords = [k for k in keywords if len(k) > 1]
        # print(keywords)
        
        for kw in keywords:
            if kw in doc_names:
                minor_id = str(doc2id[kw])
                for ind, sent in enumerate(doc):
                    if kw in sent:
                        if minor_id not in final_relations[main_doc_id].keys():
                            final_relations[main_doc_id][minor_id]=[{"kw":kw,"type":"hyper_link", "info":[ind]}]
                        else:
                            final_relations[main_doc_id][minor_id].append({"kw":kw,"type":"hyper_link", "info":[ind]})
                        if ind > len(doc):
                            print({"kw":kw,"type":"hyper_link", "info":[ind]}, len(doc),doc)
                            print("=================")
                    # break
    logger.info("Done.")
    return final_relations

def addKGRelation(docs,graph_relations):
    def depart(node_str):
        node_str = str(node_str)
        if '[' in node_str:
            k = node_str.split('[')[0]
            dest = node_str.split('[')[1].replace(']', '')
            return k, dest
        else:
            return node_str, ''
    import pandas as pd
    logger.info("Reading the knowledge graph, please wait...")
    datas = pd.read_csv('../supply/ownthink_v2.csv', encoding='utf8')
    logger.info("Done.")
    
    entitys = datas['实体']
    edges = datas['属性']
    values = datas['值']
    reduce = {}
    
    logger.info("Simplify the Knowledge Graph.")
    
    if not os.path.exists("../supply/reduced_ownthink.pkl"):
        for entity, edge, value in tqdm(zip(entitys.to_list(),edges.to_list(),values.to_list() )):
            if entity in reduce.keys():
                reduce[entity].append((edge, value))
            else:
                reduce[entity] = [(edge, value)]
        
        dump_pickle(reduce, "../supply/reduced_ownthink.pkl")
    else:
        reduce = read_pickle("../supply/reduced_ownthink.pkl")
    logger.info("Done.")
    
    to_deal_keys = set()
    for k in reduce.keys():
        if isinstance(k, float):
            continue
        query = str(k)
        if '[' in k:
            query = k.split('[')[0]
        if query in doc_names:
            to_deal_keys.add(k)
    
    logger.info("Build the Knowledge Graph relation.")
    for p_key in tqdm(list(to_deal_keys)):
        relative_nodes = reduce[p_key]
        main_doc,_ = depart(p_key)
        main_doc = str(main_doc)
        
        if doc2id[main_doc] not in graph_relations:
            graph_relations[doc2id[main_doc]]={}
            
        if main_doc not in doc_names:
            continue
        
        # print(doc2id[main_doc])
        
        if len(main_doc) < 2:
            continue
        for node in relative_nodes:
            relation = node[0]
            if relation=='歧义关系' or relation=='歧义权重' :
                continue
            end_node_str = node[1]
            query, dest = depart(end_node_str)

            
            if query not in doc_names:
                continue
            doc_id = doc2id[query]
            # print(main_doc)
            if doc_id not in graph_relations[doc2id[main_doc]].keys():
                graph_relations[doc2id[main_doc]][doc_id]=[{"type":"rela", "info":[relation]}]
            else:
                graph_relations[doc2id[main_doc]][doc_id].append({"type":"rela", "info":[relation]})
    logger.info("Done.")
    
    return graph_relations

def addWVRelation(docs,graph_relations):
    matched_keys = []
    
    def utilize_gensim():
    
        embed_path = '../supply/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'
        # 输出文件
        

        if os.path.exists(embed_path.replace(".txt", ".bin")):
            wv_from_text = gensim.models.KeyedVectors.load(embed_path.replace(".txt", ".bin"), mmap='r')
        else:
            wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=False,)
            wv_from_text.init_sims(replace=True)    # 后续可以做similarity的运算
            wv_from_text.save(embed_path.replace('.txt', '.bin'))  # convert to bin format

        vocab = wv_from_text.key_to_index
        logger.info("Vocabulary Size: %s" % len(vocab.keys()))     # 词表大小

        word_vocab = dict()
        word_vocab['PAD'] = 0
        # word_vocab['UNK'] = 1
        for key in vocab.keys():
            if key in doc2id.keys():
                matched_keys.append(key)
                word_vocab[key] = len(word_vocab.keys())
        logger.info("Vocabulary Size: %s" % len(word_vocab.keys()))

        vector_size = wv_from_text.vector_size
        logger.info("Vector size: %s" % vector_size)

        word_embed = wv_from_text.vectors
        logger.info("Embedding shape: {}".format(word_embed.shape))     # 词向量维度


        unk_embed = np.random.randn(1, 200)
        pad_embed = np.zeros(shape=(1, 200), dtype=np.float64)
        extral_embed = np.concatenate((pad_embed, unk_embed), axis=0)

        word_embed = np.concatenate((extral_embed, word_embed), axis=0)
        logger.info("Embedding shape: {}".format(word_embed.shape))

        # 保存到本地
        np.save('../supply/tencent-ailab-embedding-zh-d200-v0.2.0-s/glove_word_embedding_200d.npy', word_embed)
        pd.to_pickle(word_vocab, '../supply/tencent-ailab-embedding-zh-d200-v0.2.0-s/glove_word_vocab_200d.pkl')

        return wv_from_text

    def most_similarity(wv_from_text,words):
        most_similar = wv_from_text.most_similar(words, topn=10)
        return most_similar

    logger.info("Processing the word2vec weights.")
    wv_from_text = utilize_gensim()
    logger.info("Processing the word2vec weights. Done!")

    # for v in doc2id.values():
    #     word2vec_relation[v] = {}

    logger.info("Start to build word2vec relations in KiDG.")
    
    for key in tqdm(matched_keys):
        if len(key) < 2:
            continue
        try:
            sims = most_similarity(wv_from_text=wv_from_text,words=[key])
            have = 0
            graph_relations[doc2id[key]] = {}
            for sim in sims:
                if sim[0] in kb_kv.keys() and have < 5 and sim[1] > 0.7 :
                    if doc2id[sim[0]] not in graph_relations[doc2id[key]]:
                        graph_relations[doc2id[key]][doc2id[sim[0]]]=[{"type":"sim", "info":[]}]
                    else:
                        graph_relations[doc2id[key]][doc2id[sim[0]]].append({"type":"sim", "info":[]})
                    
                    if doc2id[key]  not in graph_relations[doc2id[sim[0]]]:
                        graph_relations[doc2id[sim[0]]][doc2id[key]]=[{"type":"sim", "info":[]}]
                    else:
                        graph_relations[doc2id[sim[0]]][doc2id[key]].append({"type":"sim", "info":[]})
                        
        except:
            continue
    logger.info("End of the buildinf of word2vec relations in KiDG.")
    
    return graph_relations

if __name__=="__main__":
    from argparse import ArgumentParser
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--intra_path', type=str, default="../data/IntraDocumentRelations/docs_F.pkl",help="path_to_intra_doc_results")
    arg_parser.add_argument('--output_path', type=str, default="../data/InterDocumentRelations", help="path_for_storing_inter_doc_results")
    
    args = arg_parser.parse_args()
    
    doc_score = read_pickle(args.intra_path)
    
    docs = [ds[0] for ds in doc_score]
    kb_kv = {v[0].strip():v[1:] for v in docs }
    id2doc = list(kb_kv.keys())
    doc2id = {k:v for v,k in enumerate(id2doc)}

    doc_keys = set(id2doc)
    doc_names = set([k  for k in kb_kv.keys()])
    
    # build hyper link relation
    Graph = None
    Graph_hyp = findHyperlink(docs)
    
    # add kg relation
    Graph_hyp_kg = addKGRelation(docs, Graph_hyp)
    
    # add word vector relation
    Graph_hyp_kg_wv = addWVRelation(docs, Graph_hyp_kg)
    
    write_json(Graph_hyp_kg_wv, f"{args.output_path}")
    logger.info(f"Results are stored in {args.output_path}")


