import json

import pickle


def read_json_origin(path):

    with open(path, encoding='utf-8') as f:
        lines = json.load(f)
    # lines = [[remove_ref(d) for d in line  ]for line in lines]

    return lines
def merge_json_origin(paths):

    datas = []
    for path in paths:
        datas += read_json_origin(path)
    return datas


def read_json(path):

    with open(path, encoding='utf-8') as f:
        lines = json.load(f)
    lines = [[remove_ref(d) for d in line  ]for line in lines]

    return lines

def write_json(data, path):

    with open(path, 'w', encoding='utf-8') as f:

        json.dump(data, f, ensure_ascii=False, indent=4)
import re

def dump_pickle(data, path):
    with open(path,'wb') as f:
        pickle.dump(data, f)
def read_pickle( path):
    with open(path,'rb') as f:
        return pickle.load( f)

def remove_ref(txt):

    ur = '\[[0-9]-?[0-9]?\]'
    bad_words = re.findall(ur, txt)
    s = txt
    # print(bad_words)
    for bw in bad_words:
        s = s.replace(bw,'')
    return s