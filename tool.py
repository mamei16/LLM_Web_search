import logging
from datetime import datetime

from modules.extensions import state


logger = logging.getLogger('text-generation-webui')

if not "LLM_Web_search" in state:
    raise ValueError("Can't use llm_web_search tool. LLM_Web_search extension not loaded.")


extension_module = state["LLM_Web_search"][-1]

params                   = extension_module.params
retrieve_from_duckduckgo = extension_module.retrieve_from_duckduckgo
retrieve_from_searxng    = extension_module.retrieve_from_searxng
Generator                = extension_module.Generator
document_retriever       = extension_module.document_retriever


tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Execute a web search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or keywords to search for."},
            },
            "required": ["query"]
        }
    }
}


def docs_to_dicts(docs):
    dicts = []
    for i, doc in enumerate(docs):
        doc_dict = {#"Result #": i+1,
                    "Content": doc.page_content,
                    "Source URL": doc.metadata['source']}
        dicts.append(doc_dict)
    return dicts


def execute(arguments):
    if not "LLM_Web_search" in state:
        error_msg = "Can't use llm_web_search tool. LLM_Web_search extension not loaded."
        logger.error(error_msg)
        return error_msg
    elif not params["enable"]:
        error_msg = "Can't use llm_web_search tool. LLM_Web_search extension is disabled."
        logger.error(error_msg)
        return error_msg

    query = arguments.get('query', '')

    max_search_results = int(params["search results per query"])
    instant_answers    = params["instant answers"]
    simple_search      = params["simple search"]
    searxng_url        = params["searxng url"]

    document_retriever.num_results                         = int(params["duckduckgo results per query"])
    document_retriever.similarity_threshold                = params["langchain similarity score threshold"]
    document_retriever.chunk_size                          = params["chunk size"]
    document_retriever.ensemble_weighting                  = params["ensemble weighting"]
    document_retriever.splade_batch_size                   = params["splade batch size"]
    document_retriever.chunking_method                     = params["chunking method"]
    document_retriever.chunker_breakpoint_threshold_amount = params["chunker breakpoint_threshold_amount"]
    document_retriever.client_timeout                      = params["client timeout"]
    document_retriever.token_classification_model_id       = params["token classification model id"]

    if searxng_url == "":
        search_generator = Generator(retrieve_from_duckduckgo(query,
                                                              document_retriever,
                                                              max_search_results,
                                                              simple_search))
    else:
        search_generator = Generator(retrieve_from_searxng(query,
                                                           searxng_url,
                                                           document_retriever,
                                                           max_search_results,
                                                           instant_answers,
                                                           simple_search))
    try:
        for _ in search_generator:
            pass
        search_results_dict = {"Results": docs_to_dicts(search_generator.retval),
                               "Current datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as exc:
        exception_message = str(exc)
        search_results_dict = {"Error": exception_message}
        logger.warning(f'LLM_Web_search | {query} generated an exception: {exception_message}')

    return search_results_dict
