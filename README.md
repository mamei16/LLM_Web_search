# Give your local LLM the ability to search the web!


## Installation

1. Install the necessary requirements found in `requirements.txt`, for example by running:  
```pip install -r requirements.txt```
2. Create a folder called `llm_web_search` in `text-generation-webui/extensions/`
2. Copy all python files to this new folder
3. Launch the Web UI with:  
```python server.py --extension llm_web_search```

If the installation was successful and the extension was loaded, a new tab with the 
title "LLM Web Search" should be visible in the web UI.

See https://github.com/oobabooga/text-generation-webui/wiki/07-%E2%80%90-Extensions for more
information about extensions.

## Usage

Currently, one a single search query per model chat message is supported.

Search queries are extracted
from the model's output using a regular expression. This is made easier by prompting the model
to use a fixed search command (see `example_instruction_templates` for example prompts). 
### Using a custom regular expression
The default regular expression is:  
```
Search_web: \"(.*)\"
```
Where `Search_web` is the search command and everything between the subsequent quotation marks
will be used as the search query. Note that every custom regular expression must use a
[capture group](https://www.regular-expressions.info/brackets.html) to extract the search
query. I recommend https://www.debuggex.com/ to try out custom regular expressions. If a regex
fulfills the requirement above, the search query should be matched by "Group 1" in Debuggex.
