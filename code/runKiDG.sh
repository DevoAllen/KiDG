path2IntraG=../data/IntraDocumentRelations
path2InterG=../data/InterDocumentRelations/kdconv_inter_relations.json
path2simcse=../supply/sentEmbed/
path2graph=../data/ResultGraph/kdconv_cross.pkl

# Building Intra-graph realations
python intraRelations.py \
--dir_path ../data/Documents/kdconv_docs.json \
--output_path ${path2IntraG} \
--simcse ${path2simcse}
# --dir_path stores the documents to be processed; 
# --output_path is the storage path for intra-graph relations; 
# --simcse is the storage path for the sentence embedding model.


# Building Inter-graph realations
python interRelations.py \
--intra_path ${path2IntraG}/kdconv_intra_relations_F.pkl \
--output_path ${path2InterG}


# Calculate the correlation between documents and pre-calculate key nodes.
python build_cross_relation.py \
--intra_path ${path2IntraG}/kdconv_intra_relations_F.pkl \
--inter_path ${path2InterG} \
--simcse ${path2simcse} \
--output_path ${path2graph}


# Traversing the KiDG to obtain the sentence sequences.
python traverseGraph.py
--intra_path ${path2IntraG}/kdconv_intra_relations_F.pkl \
--inter_path ${path2InterG} \
--graph_path ${path2graph}\
--output_path ../output
    
