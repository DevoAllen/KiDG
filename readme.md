# Retrieval-free Knowledge Injection through Multi-Document Traversal for Dialogue Models
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
## Introduction
Dialogue models are often enriched with extensive external knowledge to provide informative responses through a retrieval-augmented pipeline. 
Nevertheless, retrieval-augmented approaches rely on finely annotated retrieval training data and knowledge-grounded response generation data, making it costly to transfer.

To tackle this challenge, this paper proposed a retrieval-free approach, KiDG, by automatically turning knowledge documents into simulated multi-turn dialogues through a MultiDocument Traversal algorithm. 
The simulated knowledge-intensive dialogues constructed by KiDG in one domain can be easily used to train and enhance pre-trained dialogue models’ knowledge w.r.t. this domain without costly annotation. 

We conduct extensive experiments comparing retrieval-augmented models and a variety of retrieval-free models. We found that dialogue models enhanced with data simulated with KiDG largely outperform state-ofthe-art retrieval-free methods, and it achieves comparable performance compared to retrievalaugmented methods while being better, and cheaper at domain transfer. 

## Usage

### 1. Installation

Clone the repo and install dependent packages:

```bash
  git clone https://github.com/DevoAllen/KiDG.git
  cd KiDG
  pip install -r requirements.txt
```
### 2. Download prerequiste materials
Out of respect for these open-source projects, please download them by yourself and place them in **supply** path.

#### Download word embedding datasets
The word embedding datasets we used is [Tencent AI Lab Embedding Corpora](https://ai.tencent.com/ailab/nlp/en/download.html).
```bash
  cd supply
  mkdir word2vec
  wget https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d200-v0.2.0-s.tar.gz
```

#### Download Knowledge Graph.

The knowledge graph we used is [ownthink](https://github.com/ownthink/KnowledgeGraphData).
Please put the downloaded file under **supply/KG**.

#### Download sentence embedding model.
In this paper, we have trained a [chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large) model using [simcse](https://github.com/princeton-nlp/SimCSE) as the backbone for bert-score to enhance performance. You can download it from [sentEmbed](https://drive.google.com/file/d/1yM02yOm-CQya2maeNRdaNJQGKXkrIi5x/view?usp=drive_link) and place it in the **supply/sentEmbed** path.



### 3. Running KiDG.


```bash
  bash runKiDG.sh
```
This script will complete the construction of the KiDG graph, and traverse it to obtain sentence sequences.

#### Inpainting Model
We use [bart-chinese-large](https://huggingface.co/fnlp/bart-large-chinese) as the backbone for the inpainting model. Please collect dialogue corpora on your own. 

The training of the inpainting model is essentially the **token infilling** task of the BART model. Given a dialogue $[u_1, u_2, u_3, u_4]$, where $u_i$ is an utterance, the input is $[u_1, u_2, MASK, u_4]$, then the corresponding label is $u_3$ (Details can be found in our paper and [Dialogue Inpainting] paper(https://arxiv.org/abs/2205.09073)). 

For code reference, you can refer to the BART training [examples](https://github.com/fastnlp/CPT).


## Acknowledgement

This repo benefits from
[CPT](https://github.com/fastnlp/CPT), [SimCSE](https://github.com/princeton-nlp/SimCSE),
[ownthink](https://github.com/ownthink/KnowledgeGraphData),
[Tencent AI Lab Embedding Corpora](https://ai.tencent.com/ailab/nlp/en/download.html).

Thanks for their wonderful works!


# Citations
If you find our project helpful, hope you can star our repo and cite our paper as follows:
```
@inproceedings{wang-etal-2023-retrieval,
    title = "Retrieval-free Knowledge Injection through Multi-Document Traversal for Dialogue Models",
    author = "Wang, Rui  and
      Bao, Jianzhu  and
      Mi, Fei  and
      Chen, Yi  and
      Wang, Hongru  and
      Wang, Yasheng  and
      Li, Yitong  and
      Shang, Lifeng  and
      Wong, Kam-Fai  and
      Xu, Ruifeng",
    year = "2023",
    url = "https://aclanthology.org/2023.acl-long.364",
    doi = "10.18653/v1/2023.acl-long.364",
    pages = "6608--6619",
}
```
 
