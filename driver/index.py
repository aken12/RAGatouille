from ragatouille import RAGPretrainedModel
import os
import argparse
from tqdm import tqdm

import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def document_loader(docs_file,is_tsv=False):
    dids = []
    texts = []
    if not is_tsv:
        with open(docs_file)as fr:
            for line in fr:
                line = json.loads(line)
                did = line["id"]
                if "title" in line:
                    text = line["title"] + ' ' + line["contents"]
                else:
                    text = line["contents"]
                dids.append(did)
                texts.append(text)
    else:
        with open(docs_file)as fr:
            next(fr)
            for line in fr:
                did,text,title = line.strip().split('\t')
                text = title + ' ' + text
                text = text.strip()
                dids.append(did)
                texts.append(text)  
    return dids, texts

def write_result(output_path,scores,reranker):
    with open(output_path,"w")as fw:
        for qid, ranked_results in scores.items():
            for rank,result in enumerate(ranked_results,start=1):
                fw.write(f'{qid} 0 {result.doc_id} {rank} {result.score} {reranker}\n')

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir',help='',default='/home/ace14788tj/extdisk/data/',type=str)
    parser.add_argument('--task',default="nq",type=str)
    parser.add_argument('--rewrite',default="default",type=str)
    parser.add_argument('--model_name',default="colbert-ir/colbertv2.0",type=str)
    parser.add_argument('--doc_max_length',default=256,type=int)
    
    args = parser.parse_args()

    model_name = args.model_name

    task = args.task
    rewrite = args.rewrite
    model_name = args.model_name
    
    if "colbertv2.0" in model_name:
        retriever = "colbertv2.0"
    else:
        retriever = "colbertv2.0"
    
    # hard coding for debugging
    # corpus_dir = os.path.join(args.data_dir,task,"corpus")
    # corpus_dir = "/home/ace14788tj/extdisk/data/wikipedia_2020/data/wikipedia_split/"
    # corpus_dir = "/home/ace14788tj/extdisk/data/wikipedia/data/collections/"
    corpus_dir = "/home/ace14788tj/extdisk/data/clueweb09/corpus_jsonl_bs4/"

    RAG = RAGPretrainedModel.from_pretrained('/home/ace14788tj/aken12_2/RAGatouille/models/models--colbert-ir--colbertv2.0/snapshots/c1e84128e85ef755c096a95bdb06b47793b13acf')
    
    logger.info('corpus loading start!!!')
    
    corpus = []
    doc_ids = []
    
    passage_num = 0
    
    for i,f in enumerate(os.listdir(corpus_dir),start=1):
        if f.endswith('.jsonl'):
            corpus_path = os.path.join(corpus_dir, f)
            dids, texts = document_loader(corpus_path,is_tsv=False)
            corpus.extend(texts)
            doc_ids.extend(dids)
            passage_num += len(texts)
            
            # corpus = corpus[:300000]
            # doc_ids = doc_ids[:300000]
            # break
          
        logger.info(f'we have {passage_num} passages...')
        logger.info('indexing start!!!')
        
        if i == 1:
            RAG.index(
                collection=corpus, 
                document_ids=doc_ids,
                index_name=f'{task}', 
                max_document_length=args.doc_max_length, 
                split_documents=False
                )
        
        else:
            RAG.add_to_index(
                new_collection=corpus, 
                new_document_ids=doc_ids,
                index_name=f'{task}', 
                split_documents=False
                )

if __name__=="__main__":
    main()