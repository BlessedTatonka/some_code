import random

from datasets import load_dataset
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator,
)


def ir_evaluate(use_instructions=False, query_prompt="search_query: ", passage_prompt="search_document: "):
    """
    Loads the ai-forever/rubq-retrieval dataset for IR evaluation.
    The corpus is reduced to relevant docs plus a random sample.
    Returns an InformationRetrievalEvaluator.
    """
    corpus = load_dataset("ai-forever/rubq-retrieval", "corpus", split="corpus")
    queries = load_dataset("ai-forever/rubq-retrieval", "queries", split="queries")
    relevant_docs_data = load_dataset("ai-forever/rubq-retrieval", split="test")

    if not use_instructions:
        query_prompt = ""
        passage_prompt = ""

    corpus = corpus.map(lambda x: {"text": x["title"] + " " + x["text"]}, remove_columns=["title"])
    required_corpus_ids = set(map(str, relevant_docs_data["corpus-id"]))
    required_corpus_ids |= set(
        random.sample(corpus["_id"], k=56826)
    )  # k can be lowered, but no need. 56826 is full dataset.
    corpus = corpus.filter(lambda x: x["_id"] in required_corpus_ids)
    corpus_dict = dict(zip(corpus["_id"], corpus["text"]))
    queries_dict = dict(zip(queries["_id"], queries["text"]))
    relevant_docs = {}
    for qid, cid in zip(relevant_docs_data["query-id"], relevant_docs_data["corpus-id"]):
        qid_str, cid_str = str(qid), str(cid)
        if qid_str not in relevant_docs:
            relevant_docs[qid_str] = set()
        relevant_docs[qid_str].add(cid_str)
    return InformationRetrievalEvaluator(
        queries=queries_dict,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name="IR",
        write_csv=False,
        query_prompt=query_prompt,
        corpus_prompt=passage_prompt,
    )


def sts_evaluate():
    stsb_eval_dataset = load_dataset("ai-forever/ru-stsbenchmark-sts", split="validation")

    return EmbeddingSimilarityEvaluator(
        sentences1=stsb_eval_dataset["sentence1"],
        sentences2=stsb_eval_dataset["sentence2"],
        scores=stsb_eval_dataset["score"],
        name="STS",
        write_csv=False,
    )
