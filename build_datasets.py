import random 
import csv

def build_dataset(uds, querystr, outfile):
    """
    Args:
        (UDSCorpus) uds,
        (str) querystr: the string for positive examples
        (str) outfile: the name of the file to save the dataset to
    """
    results = uds.query(querystr, query_type='edge', cache_rdf=False)
    dataset = []
    pos_cnt = 0
    neg_cnt = 0
    for gid, edges in results.items():
        tokenized_sentence = uds[gid].sentence

        if not edges: #create a negative example
            label = 0
            sem_edges = dict((x,y) for x, y in uds[gid].semantics_edges().items() if y['type'] == 'dependency')
            sem_edges= list(sem_edges)
            sem_edges= [x for x in sem_edges if 'root' not in x[0]]
            #choose a random edge
            if not sem_edges:
                continue
            picked_edge = random.choice(sem_edges)
            pred_id = picked_edge[0].replace("semantics-pred", "syntax")
            arg_id = picked_edge[1].replace("semantics-arg", "syntax")
            pred_head = uds[gid].nodes[pred_id]['lemma']
            arg_head = uds[gid].nodes[arg_id]['lemma']
            if (pred_id, arg_id) in uds[gid].edges:
                dependency = uds[gid].edges[(pred_id, arg_id)]['deprel']
            else:
                dependency = "NONE"

            dataset += [{"sentence":tokenized_sentence, "pred_head":pred_head, "arg_head":arg_head, "dep":dependency, "label":label}]
            neg_cnt +=1

        else: 
            label = 1
            for edge, data in edges.items():
                pred_id = edge[0].replace("semantics-pred", "syntax")
                arg_id = edge[1].replace("semantics-arg", "syntax")
                pred_head = uds[gid].nodes[pred_id]['lemma']
                arg_head = uds[gid].nodes[arg_id]['lemma']
                if (pred_id, arg_id) in uds[gid].edges:
                    dependency = uds[gid].edges[(pred_id, arg_id)]['deprel']
                else:
                    dependency = "NONE"
                dataset += [{"sentence":tokenized_sentence, "pred_head":pred_head, "arg_head":arg_head, "dep":dependency, "label":label}]
                pos_cnt += 1

    with open(outfile, 'w') as fi:
        fields = ['sentence', 'pred_head', 'arg_head', 'dep', 'label']
        writer = csv.DictWriter(fi, fieldnames=fields)
        writer.writeheader()
        for i in dataset:
            writer.writerow(i)

    return pos_cnt, neg_cnt

    
    
