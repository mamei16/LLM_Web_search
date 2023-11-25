import time
import re

import gradio as gr

import modules.shared as shared
from modules.logging_colors import logger
from modules.text_generation import generate_reply_HF, generate_reply_custom
from .llm_web_search import search_duckduckgo, dict_list_to_pretty_str

params = {
    "display_name": "LLM Web Search",
    "Enable": True,
    "is_tab": True,
    "show search replies": True,
    "top search replies per query": 5
}


def setup():
    """
    Is executed when the extension gets imported.
    :return:
    """
    pass


def ui():
    """
    Creates custom gradio elements when the UI is launched.
    :return:
    """
    with gr.Row():
        enable = gr.Checkbox(value=params['Enable'], label='Enable LLM web search')

    enable.change(lambda x: params.update({"Enable": x}), enable, None)


def custom_generate_reply(question, original_question, seed, state, stopping_strings, is_chat):
    """
    Overrides the main text generation function.
    :return:
    """
    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        yield ''
        return

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel', 'ExllamaModel', 'Exllamav2Model',
                                           'CtransformersModel']:
        generate_func = generate_reply_custom
    else:
        generate_func = generate_reply_HF

    search_error_message = None
    matched_patterns = {}
    max_search_results = params["top search replies per query"]
    for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
        search_re_match = re.search(f"Search_web: \".*\"", reply)
        if search_re_match is not None:
            matched_pattern = search_re_match.group(0)
            if matched_patterns.get(matched_pattern):
                continue
            matched_patterns[matched_pattern] = True
            search_term = matched_pattern.split(" ", 1)[1].replace("\"", "").rstrip("end")
            print(f"Searching for {search_term}...")
            try:
                search_results = search_duckduckgo(search_term, max_search_results)
            except Exception as exc:
                exception_message = str(exc)
                search_error_message = f"The search tool encountered an error: {exception_message}"
                print(f'{search_term} generated an exception: {exception_message}')

            reply += "\n```"
            reply += "\nSearch tool:\n"
            yield reply
            if search_error_message:
                reply += search_error_message
            else:
                reply += dict_list_to_pretty_str(search_results)
            yield reply
            time.sleep(0.041666666666666664)
            reply += "```"
            yield reply

        yield reply


def output_modifier(string, state, is_chat=False):
    """
    Modifies the output string before it is presented in the UI. In chat mode,
    it is applied to the bot's reply. Otherwise, it is applied to the entire
    output.
    :param string:
    :param state:
    :param is_chat:
    :return:
    """
    return string


def custom_css():
    """
    Returns custom CSS as a string. It is applied whenever the web UI is loaded.
    :return:
    """
    return ''


def custom_js():
    """
    Returns custom javascript as a string. It is applied whenever the web UI is
    loaded.
    :return:
    """
    return ''


def chat_input_modifier(text, visible_text, state):
    """
    Modifies both the visible and internal inputs in chat mode. Can be used to
    hijack the chat input with custom content.
    :param text:
    :param visible_text:
    :param state:
    :return:
    """
    return text, visible_text


def state_modifier(state):
    """
    Modifies the dictionary containing the UI input parameters before it is
    used by the text generation functions.
    :param state:
    :return:
    """
    return state


def history_modifier(history):
    """
    Modifies the chat history before the text generation in chat mode begins.
    :param history:
    :return:
    """
    return history
