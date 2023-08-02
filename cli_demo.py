from configs.model_config import *
from chains.local_doc_qa import LocalDocQA
import os
import nltk
from models.loader.args import parser
import models.shared as shared
from models.loader import LoaderCheckPoint
import sys
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

def qa_main(file_index_path, submit_file):
    # llm_model_ins = None
    llm_model_ins = shared.loaderLLM()
    llm_model_ins.history_len = LLM_HISTORY_LEN

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=llm_model_ins,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          top_k=VECTOR_SEARCH_TOP_K)
    import jsonlines
    from pypinyin import lazy_pinyin
    name_set = {}
    with open('name.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            cl = line.split('__')
            fn = line.replace('.pdf', '')
            v = name_set.get(cl[1],[])
            v.append(fn)
            name_set[cl[1]] = v
            v = name_set.get(cl[-3],[])
            v.append(fn)
            name_set[cl[-3]] = v
    
    logger.info(f"debug name set:{len(name_set)}")
    with jsonlines.open("submit.jsonl", mode='w') as sjl:
        lines = []
        with jsonlines.open(submit_file, 'r') as jl:
            for data in jl:
                query = data['question']
                data['answer'] = "信息不足，无法回答。"
                lines.append(data)
                try:
                    vs_paths = get_name_set_path(name_set, query)
                    logger.info(f"from name set get vs_path:{vs_paths}")
                    if len(vs_paths) == 0:
                        logger.info(f"query:{query} not find pdf")
                        continue;
                    index_vs_paths = []
                    for vs_path in vs_paths:
                        pinyin_path = "".join(lazy_pinyin(vs_path))
                        index_vs_path = os.path.join('/home/zh.wang/chatglm_llm_fintech_raw_dataset/knowledge_base/index', pinyin_path).strip()
                        new_index = os.path.join('/share-dc/zh.wang/index', pinyin_path).strip()
                        is_not_exists = not os.path.exists(index_vs_path)
                        logger.info(f'load index path:{index_vs_path} not exist:{is_not_exists}')
                        if is_not_exists:
                            index_vs_path = new_index
                            if not os.path.exists(index_vs_path):
                                logger.info(f'load new index path:{index_vs_path}')
                                pdf_vs_path = vs_path if "pdf" in vs_path else f"{vs_path}.pdf"
                                file = os.path.join('/home/zh.wang/fintech_raw_dataset/chatglm_llm_fintech_raw_dataset/allpdf', pdf_vs_path)
                                k, _ = local_doc_qa.init_knowledge_vector_store(file, vs_path=index_vs_path)
                                logging.info(f"index path:{k}")
                                if not k:
                                    logging.info(f"not create {file} faiss index")
                                    continue
                        else:
                            logger.info(f"already create index path:{index_vs_path}")
                        index_vs_paths.append(index_vs_path)
                    for resp, _ in local_doc_qa.get_knowledge_based_answer_list(query=query, paths=index_vs_paths):
                            
                        r = resp['result']
                        logger.info(f"id:{data['id']}\tanwser:{r}")
                        data['answer'] = r
                        if REPLY_WITH_SOURCE:
                            source_text = [f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                                        # f"""相关度：{doc.metadata['score']}\n\n"""
                                        for inum, doc in
                                        enumerate(resp["source_documents"])]
                            logger.info("\n\n" + "\n\n".join(source_text))
                except:
                    import traceback
                    traceback.print_exc()
                    continue
        sjl.write_all(lines)

def get_name_set_path(name_set, query):
    vs_path = set()
    for k in name_set:
        if k in query:
            fs = name_set[k]
            for f in fs:
                year = f.split('__')[-2]
                year = year.replace('年', '')
                dy = int(year)
                if year in query or str(dy+1) in query:
                    vs_path.add(f)
                    continue
    return vs_path


def main(gpus, index, device_num, debug):
    if index >= gpus:
        logger.info(f"index:{index} out of gpus:{gpus}")
        return
    local_doc_qa = LocalDocQA()
    device = index % device_num
    logging.info(f"load embedding to {EMBEDDING_DEVICE}:{device}")
    local_doc_qa.init_cfg(llm_model=None,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=f"{EMBEDDING_DEVICE}:{device}",
                          top_k=VECTOR_SEARCH_TOP_K)
    filepath = '/home/zh.wang/fintech_raw_dataset/chatglm_llm_fintech_raw_dataset/allpdf'
    import glob, os
    pdf_files = glob.glob(os.path.join(filepath, "*.pdf"))
    local_doc_qa.init_file_knowledge_index("/home/zh.wang/fintech_raw_dataset/index",pdf_files)
    return
    vs_path = None
    while not vs_path:
        # print("注意输入的路径是完整的文件路径，例如knowledge_base/`knowledge_base_id`/content/file.md，多个路径用英文逗号分割")
        # filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
        filepath = '/home/zh.wang/chatglm_llm_fintech_raw_dataset/allpdf'
        # 判断 filepath 是否为空，如果为空的话，重新让用户输入,防止用户误触回车
        if not filepath:
            continue
        import glob, os
        pdf_files = glob.glob(os.path.join(filepath, "*.pdf"))
        # pdf_files.reverse()
        SPLIT_LEN = int(len(pdf_files) / gpus)
        sub_lists = [pdf_files[i:i+SPLIT_LEN] for i in range(0, len(pdf_files), SPLIT_LEN)]
        pdf_files = sub_lists[int(index)]
        if debug:
            print(f"load pdf index:{index}, split len:{SPLIT_LEN}")
            return
        # 支持加载多个文件
        # filepath = filepath.split(",")
        # filepath错误的返回为None, 如果直接用原先的vs_path,_ = local_doc_qa.init_knowledge_vector_store(filepath)
        # 会直接导致TypeError: cannot unpack non-iterable NoneType object而使得程序直接退出
        # 因此需要先加一层判断，保证程序能继续运行
        temp,_ = local_doc_qa.init_knowledge_vector_store(pdf_files)
        if temp is not None:
            vs_path = temp
            # 如果loaded_files和len(filepath)不一致，则说明部分文件没有加载成功
            # 如果是路径错误，则应该支持重新加载
            # if len(loaded_files) != len(filepath):
            #     reload_flag = eval(input("部分文件加载失败，若提示路径不存在，可重新加载，是否重新加载，输入True或False: "))
            #     if reload_flag:
            #         vs_path = None
            #         continue

            print(f"the loaded vs_path is 加载的vs_path为: {vs_path}")
        else:
            print("load file failed, re-input your local knowledge file path 请重新输入本地知识文件路径")
    return
    history = []
    while True:
        query = input("Input your question 请输入问题：")
        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,
                                                                     streaming=STREAMING):
            if STREAMING:
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
            else:
                print(resp["result"])
        if REPLY_WITH_SOURCE:
            source_text = [f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                           # f"""相关度：{doc.metadata['score']}\n\n"""
                           for inum, doc in
                           enumerate(resp["source_documents"])]
            print("\n\n" + "\n\n".join(source_text))


if __name__ == "__main__":
#     # 通过cli.py调用cli_demo时需要在cli.py里初始化模型，否则会报错：
    # langchain-ChatGLM: error: unrecognized arguments: start cli
    # 为此需要先将
    # args = None
    # args = parser.parse_args()
    # args_dict = vars(args)
    # shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    # 语句从main函数里取出放到函数外部
    # 然后在cli.py里初始化
    # args = None
    # args = parser.parse_args()
    # args_dict = vars(args)
    # shared.loaderCheckPoint = LoaderCheckPoint(args_dict)
    # main()
    name_set = {}
    with open('name.txt', 'r') as f:
        for line in f:
            line = line.strip('\n')
            cl = line.split('__')
            fn = line.replace('.pdf', '')
            v = name_set.get(cl[1],[])
            v.append(fn)
            name_set[cl[1]] = v
            v = name_set.get(cl[-3],[])
            v.append(fn)
            name_set[cl[-3]] = v
    vs = get_name_set_path(name_set=name_set, query='请问，光云科技2019年的财务费用是多少元?')
    logging.info(f"recall path:{vs}")