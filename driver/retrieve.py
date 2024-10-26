from ragatouille import RAGPretrainedModel
import os
import argparse
from tqdm import tqdm

import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_loader(query_file,use_pseudo_doc=False):
    logger.info("loading query with pseudo_doc ...")
    qids = []
    queries = []
    with open(query_file)as fr:
        query_json = json.load(fr)
        for line in query_json:
            qid = line["query_id"]
            query = line["query"]
            if use_pseudo_doc:
                pseudo_doc = line["pseudo_doc"]
                query = f'{query} {pseudo_doc}'.strip()

            qids.append(qid)
            queries.append(query)
    return qids, queries

def write_result(output_path,results,qids,retriever):
    with open(output_path,"w")as fw:
        for qid,result in zip(qids,results):
            for one_result in result:
                fw.write(f'{qid} 0 {one_result["document_id"]} {one_result["rank"]} {one_result["score"]} {retriever}\n')

def args_write(args_dict,output_path="parameter.json"):
    with open(output_path,"w")as fw:
        json.dump(args_dict,fw,ensure_ascii=False,indent=4)
        
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir',help='',default='/home/ace14788tj/extdisk/data',type=str)
    parser.add_argument('--task',default="nq",type=str)
    parser.add_argument('--corpus',default="wikipedia",type=str)
    parser.add_argument('--rewrite',default="default",type=str)
    parser.add_argument('--topk',default=100,type=int)
    parser.add_argument('--model_name',default="colbert-ir/colbertv2.0",type=str)
    parser.add_argument('--eval',action='store_true')
    args = parser.parse_args()

    model_name = args.model_name

    task = args.task
    corpus = args.corpus
    rewrite = args.rewrite
    model_name = args.model_name
    topk = args.topk
    use_pseudo_doc = False
    
    if not (rewrite == "default" or rewrite == "ideal"):
        use_pseudo_doc = True
        
    if "colbertv2.0" in model_name:
        retriever = "colbertv2.0"
    else:
        retriever = "colbertv2.0"
    
    # hard coding for debugging
    # index_dir = os.path.join(args.data_dir,task,"corpus")
    index_dir = f"/home/ace14788tj/aken12_2/RAGatouille/driver/.ragatouille/colbert/indexes/{corpus}"
    query_path = os.path.join(args.data_dir,task,f"query/{rewrite}_query.json")
    qrel_path = os.path.join(args.data_dir,task,"qrels","qrel.tsv")
    output_path = os.path.join(args.data_dir,task,"runs",retriever,f'{task}_{retriever}_{rewrite}.txt')

    args_dict = {
        "index_dir": index_dir,
        "query_path": query_path,
        "output_path": output_path,
        "use_pseudo_doc": use_pseudo_doc,
        "rewrite": rewrite,
        "corpus": corpus,
        "topk": topk
    }
    
    args_write(args_dict)

    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    RAG = RAGPretrainedModel.from_pretrained('/home/ace14788tj/aken12_2/RAGatouille/models/models--colbert-ir--colbertv2.0/snapshots/c1e84128e85ef755c096a95bdb06b47793b13acf')
    RAG = RAGPretrainedModel.from_index(index_dir)
    
    qids,queries = query_loader(query_path,use_pseudo_doc)
    all_results = RAG.search(query=queries, k=topk)

    write_result(output_path,all_results,qids,retriever)

    # if args.eval:
    #     qrel_path

if __name__=="__main__":
    main()