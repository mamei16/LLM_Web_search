# Give your local LLM the ability to search the web!
![unit tests](https://github.com/mamei16/LLM_Web_search/actions/workflows/unit_tests.yml/badge.svg?branch=main)

This project gives local LLMs the ability to search the web by outputting a specific
command. Once the command has been found in the model output using a regular expression, a web search is issued, returning a number of result pages. Finally, an
ensemble of a dense embedding model and 
[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (Or alternatively, [SPLADE](https://github.com/naver/splade))
is used to extract the relevant parts (if any) of each web page in the search results
and the results are appended to the model's output.
![llm_websearch](https://github.com/mamei16/LLM_Web_search/assets/25900898/f9d2d83c-e3cf-4f69-91c2-e9c3fe0b7d89)


* **[Table of Contents](#table-of-contents)**
  * [Installation](#installation)
  * [Usage](#usage)
    + [Using a custom regular expression](#using-a-custom-regular-expression)
    + [Reading web pages](#reading-web-pages)
  * [Search backends](#search-backends)
    + [DuckDuckGo](#duckduckgo)
    + [SearXNG](#searxng)
      + [Search parameters](#search-parameters)
  * [Search types](#search-types)
	  + [Simple search](#simple-search)
	  + [Full search](#full-search)
  * [Keyword retrievers](#keyword-retrievers)
    + [Okapi BM25](#okapi-bm25)
    + [SPLADE](#splade)
  * [Chunking Methods](#chunking-methods)
    + [Character-based Chunking](#character-based-chunking)
    + [Semantic Chunking](#semantic-chunking)
    + [Token Classification based Chunking](#token-classification-based-chunking)
  * [Recommended models](#recommended-models)

## Installation
1. Go to the "Session" tab of the web UI and use "Install or update an extension" 
to download the latest code for this extension.
2. Run the appropriate `update_wizard` script inside the text-generation-webui folder
   and choose `Install/update extensions requirements`, then choose the name of this extension.
3. Launch the Web UI by running the appropriate `start` script and enable the extension under the session tab.  
 Alternatively,
you can also start the server directly using the following command (assuming you have activated your conda/virtual environment):  
```python server.py --extension LLM_Web_search```

If the installation was successful and the extension was loaded, a new tab with the 
title "LLM Web Search" should be visible in the web UI.

See https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions for more
information about extensions.


## Usage

Search queries are extracted from the model's output using a regular expression. This is made easier by prompting the model
to use a fixed search command (see `system_prompts/` for example prompts).

An example workflow of using this extension could be:
1. Load a model
2. Head over to the "LLM Web search" tab
3. Load a custom system message/prompt
4. Ensure that the query part of the command mentioned in the system message 
can be matched using the current "Search command regex string" 
(see "Using a custom regular expression" below)
5. Pick a generation parameter preset that works well for you. You can read more about generation parameters [here](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#generation)
6. Choose "chat-instruct" or "instruct" mode and start chatting

### Using a custom regular expression
The default regular expression is:  
```regexp
Search_web\("(.*)"\)
```
Where `Search_web` is the search command and everything between the quotation marks
inside the parentheses will be used as the search query. Every custom regular expression must use a
[capture group](https://www.regular-expressions.info/brackets.html) to extract the search
query. I recommend https://www.debuggex.com/ to try out custom regular expressions. If a regex
fulfills the requirement above, the search query should be matched by "Group 1" in Debuggex.

Here is an example of a more flexible, but more complex, regex that works for several
different models:
```regexp
[Ss]earch_web\((?:["'])(.*)(?:["'])\)
```
### Reading web pages
Basic support exists for extracting the full text content from a webpage. The default regex to use this
functionality is:
```regexp
Download_webpage\("(.*)"\)
```
**Note**: The full content of a web page is likely to exceed the maximum context length of your average local LLM.
## Search backends

### DuckDuckGo
This is the default web search backend. 

### SearXNG

To use a local or remote SearXNG instance instead of DuckDuckGo, simply paste the URL into the 
"SearXNG URL" text field of the "LLM Web Search" settings tab (be sure to include `http://` or `https://`). The instance must support
returning results in JSON format.

#### Search parameters
To modify the categories, engines, languages etc. that should be used for a
specific query, it must follow the
[SearXNG search syntax](https://docs.searxng.org/user/search-syntax.html). Currently, 
automatic redirect and Special Queries are not supported.


## Search types
### Simple search
Quickly finds answers using just the highlighted snippets from websites returned by the search engine. If you simply want results *fast*, choose this search type.  
Note: Some advanced options in the UI will be hidden when simple search is enabled, as they have no effect in this case.  
Note2: The snippets returned by SearXNG are often much more useful than those returned by DuckDuckGo, so consider using SearXNG as the search backend if you use simple search.
### Full search
Scans entire websites in the results for a more comprehensive search. Ideally, this search type should be able to find "needle in the haystack" information hidden somewhere in the website text. Hence, choose this option if you want to trade a more resource intensive search process for generally more relevant search results.   
For the best possible search results, also enable semantic chunking and use SPLADE as the keyword retriever.
## Keyword retrievers
### Okapi BM25
This extension comes out of the box with 
[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) enabled, which is widely used and very popular
for keyword based document retrieval. It runs on the CPU and,
for the purpose of this extension, it is fast.  
### SPLADE
If you don't run the extension in "CPU only" mode and have some VRAM to spare,
you can also select [SPLADE](https://github.com/naver/splade) in the "Advanced settings" section
as an alternative. It has been [shown](https://arxiv.org/pdf/2207.03834.pdf) to outperform BM25 in multiple benchmarks 
and uses a technique called "query expansion" to add additional contextually relevant words
to the original query. However, it is slower than BM25. You can read more about it [here](https://www.pinecone.io/learn/splade/).  

To improve performance, documents are embedded in batches and in parallel. Increasing the
"SPLADE batch size" parameter setting improves performance up to a certain point,
but VRAM usage ramps up quickly with increasing batch size. A batch size of 8 appears 
to be a good trade-off, but the default value is 2 to avoid running out of memory on smaller
GPUs.

## Chunking Methods

### Character-based Chunking

Naively partitions a website's text into fixed sized chunks without any regard for the text content. This is the default, since it is fast and requires no GPU.

### Semantic Chunking

Tries to partition a website's text into chunks based on semantics. If two consecutive sentences have very different embeddings (based on the cosine distance between their embeddings), a new chunk will be started. How different two consecutive sentences have to be for them to end up in different chunks can be tuned using the ` sentence split threshold` parameter in the UI.  
For natural language, this method generally produces much better results than character-based chunking. However, it is noticeably slower, even when using the GPU.

### Token Classification based Chunking

This chunking method employs a fine-tune of the DistilBERT transformer model, which has been trained to classify tokens (see [chonky](https://github.com/mirth/chonky)). If a token is classified as the positive class, a new paragraph (or a new chunk) is meant to be started after the token.
 
While semantic chunking only compares pairs of consecutive sentences when deciding on where to start a new chunk, the token classification model can utilize a much longer context. However, the need to process this context means that this chunking method is slower than semantic chunking.

## Recommended models
If you (like me) have ≤ 12 GB VRAM, I recommend using one of:
- [Llama-3.1-8B-instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
- [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- [Gemma-3-it](https://huggingface.co/google/gemma-3-12b-it)
- [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)  
 Since the Qwen3 family consists of reasoning models, some unique problems arise:  
  1. It seems that Qwen3 models are harder to prompt to use the search command. I have uploaded the system prompt that has worked most reliably under the name "reasoning_enforce_search".
  2. By ticking the checkbox "Enable thinking after searching" in the extension's settings, the model will resume thinking after each search. However, the main webUI only expects the model to think *once* at the start of the message, and so only the first thinking output will be put into a collapsible UI block. You can download a patch [here](https://gist.github.com/mamei16/bdcb994f93f7b3d2c389c04d32bc68d4) that fixes this. Download and extract it, then navigate to your `text-generation-webui` directory, put the patch file there and finally run `git apply ooba_multi_thinking.patch`
