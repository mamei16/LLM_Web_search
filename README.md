# Give your local LLM the ability to search the web!
This project gives local LLMs the ability to search the web by outputting a specific
command. Once the command has found in the model output using a regular expression, [duckduckgo-search](https://pypi.org/project/duckduckgo-search/)
is used to search the web and return a number of result pages. Finally, LangChain's [Contextual compression](https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/)
is used to extract the relevant parts (if any) of each web page in the search results and the results are appended to the model's
output.
![llm_websearch](https://github.com/mamei16/LLM_Web_search/assets/25900898/f9d2d83c-e3cf-4f69-91c2-e9c3fe0b7d89)
## Installation

1. Update the conda environment in which you installed the dependencies of 
[oobabooga's text-generation-webui](https://github.com/oobabooga/text-generation-webui),
by running `conda env update -n <your_environment> --file environment.yml`

2. Create a new folder inside `text-generation-webui/extensions/` and name it `llm_web_search` 
2. Copy all python files from this project into the new folder
3. Launch the Web UI with:  
```python server.py --extension llm_web_search```

If the installation was successful and the extension was loaded, a new tab with the 
title "LLM Web Search" should be visible in the web UI.

See https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions for more
information about extensions.

## Usage

Search queries are extracted from the model's output using a regular expression. This is made easier by prompting the model
to use a fixed search command (see `example_instruction_templates/` for example prompts). To get
 the best search results, queries should be specific and consist of multiple words. For example,
instead of searching for the keyword "bimgus", a better query would be "what is bimgus?".   
Currently, only a single search query per model chat message is supported.

### Using a custom regular expression
The default regular expression is:  
```regexp
Search_web: "(.*)"
```
Where `Search_web` is the search command and everything between the subsequent quotation marks
will be used as the search query. Note that every custom regular expression must use a
[capture group](https://www.regular-expressions.info/brackets.html) to extract the search
query. I recommend https://www.debuggex.com/ to try out custom regular expressions. If a regex
fulfills the requirement above, the search query should be matched by "Group 1" in Debuggex.

Here is an example of a more flexible, but more complex, regex that works for several
different models:
```regexp
Search_web: ?(?:["'])(.*)(?:["'])
```
### Reading web pages
Experimental support exists for extracting the full text content from a webpage. The default regex to use this
functionality is:
```regexp
Open_url: "(.*)"
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

### Search parameters
To modify the categories, engines, languages etc. that should be used for a
specific query, it must follow the
[SearXNG search syntax](https://docs.searxng.org/user/search-syntax.html). Currently, 
automatic redirect and Special Queries are not supported.