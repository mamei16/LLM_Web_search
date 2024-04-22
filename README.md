# Give your local LLM the ability to search the web!
This project gives local LLMs the ability to search the web by outputting a specific
command. Once the command has been found in the model output using a regular expression,
[duckduckgo-search](https://pypi.org/project/duckduckgo-search/)
is used to search the web and return a number of result pages. Finally, an
ensemble of LangChain's [Contextual compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/) and 
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
  * [Keyword retrievers](#keyword-retrievers)
    + [Okapi BM25](#okapi-bm25)
    + [SPLADE](#splade)
  * [Recommended models](#recommended-models)

## Installation
1. Go to the "Session" tab of the web UI and use "Install or update an extension" 
to download the latest code for this extension.
2. To install the extension's depencies you have two options:  
   1. **The easy way:** Run the appropriate `update_wizard` script inside the text-generation-webui folder
   and choose `Install/update extensions requirements`. This installs everything using `pip`,
   which means using the unofficial `faiss-cpu` package. Therefore, it is not guaranteed to
   work with your system (see [the official disclaimer](https://github.com/facebookresearch/faiss/wiki/Installing-Faiss#why-dont-you-support-installing-via-xxx-)).
   2. **The safe way:** Manually update the conda environment in which you installed the dependencies of 
   [oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui).
   Open the subfolder `text-generation-webui/extensions/LLM_Web_search` in a terminal or conda shell.
   If you used the one-click install method, run the command 
   `conda env update -p <path_to_your_environment> --file environment.yml`,
   where you need to replace `<path_to_your_environment>` with the path to the 
   `/installer_files/env` subfolder within the text-generation-webui folder.
   Otherwise, if you made your own environment, 
   use `conda env update -n <name_of_your_environment> --file environment.yml`  
     (NB: Solving the environment can take a while)
3. Launch the Web UI with:  
```python server.py --extension LLM_Web_search```

If the installation was successful and the extension was loaded, a new tab with the 
title "LLM Web Search" should be visible in the web UI.

See https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions for more
information about extensions.

## Usage

Search queries are extracted from the model's output using a regular expression. This is made easier by prompting the model
to use a fixed search command (see `system_prompts/` for example prompts).   
Currently, only a single search query per model chat message is supported.

An example workflow of using this extension could be:
1. Load a model
2. Load a matching instruction template
3. Head over to the "LLM Web search" tab
4. Load a custom system message/prompt
5. Ensure that the query part of the command mentioned in the system message 
can be matched using the current "Search command regex string" 
(see "Using a custom regular expression" below)
6. Pick a hyperparameter generation preset that works well for you.
7. Choose "chat-instruct" or "instruct" mode and start chatting

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
Experimental support exists for extracting the full text content from a webpage. The default regex to use this
functionality is:
```regexp
Open_url\("(.*)"\)
```
**Note**: The full content of a web page is likely to exceed the maximum context length of your average local LLM.
## Search backends

### DuckDuckGo
This is the default web search backend.

### SearXNG

Rudimentary support exists for SearXNG. To use a local or remote 
SearXNG instance instead of DuckDuckGo, simply paste the URL into the 
"SearXNG URL" text field of the "LLM Web Search" settings tab. The instance must support
returning results in JSON format.

#### Search parameters
To modify the categories, engines, languages etc. that should be used for a
specific query, it must follow the
[SearXNG search syntax](https://docs.searxng.org/user/search-syntax.html). Currently, 
automatic redirect and Special Queries are not supported.


## Keyword retrievers
### Okapi BM25
This extension comes out of the box with 
[Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) enabled, which is widely used and very popuplar
for keyword based document retrieval. It runs on the CPU and,
for the purpose of this extension, it is fast.  
### SPLADE
If you don't run the extension in "CPU only" mode and have some VRAM to spare,
you can also select [SPLADE](https://github.com/naver/splade) in the "Advanced settings" section
as an alternative. It has been [shown](https://arxiv.org/pdf/2207.03834.pdf) to outperform BM25 in multiple benchmarks 
and uses a technique called "query expansion" to add additional contextually relevant words
to the original query. You can read more about it [here](https://www.pinecone.io/learn/splade/).  
To use SPLADE, you have to install the additional dependency [qdrant-client](https://github.com/qdrant/qdrant-client). 
Simply activate the conda environment of the main web UI and run
`pip install qdrant-client`.  
To improve performance, documents are embedded in batches and in parallel. Increasing the
"SPLADE batch size" parameter setting improves performance up to a certain point,
but VRAM usage ramps up quickly with increasing batch size. A batch size of 8 appears 
to be a good trade-off, but the default value is 2 to avoid running out of memory on smaller
GPUs.

## Recommended models
If you (like me) have ≤ 12 GB VRAM, I recommend using 
[Llama-3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
You can find a matching instruction template in the extension's `instruction_templates`
folder. Simply copy it to the main web UI's `instruction-templates` folder.
**Note:** Several existing GGUF versions have a stop token issue, which can be solved by [editing the file's 
metadata](https://www.reddit.com/r/LocalLLaMA/comments/1c7dkxh/tutorial_how_to_make_llama3instruct_ggufs_less/). A GGUF version where this issue has already been fixed can be found 
[here](https://huggingface.co/AI-Engine/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q5_k_m_with_temp_stop_token_fix.gguf).
