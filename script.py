import time
import re
import json
import os
from datetime import datetime
from collections import defaultdict
from functools import partial

import gradio as gr
import torch

import modules.shared as shared
from modules import chat, ui as ui_module
from modules.utils import gradio
from modules.text_generation import generate_reply_HF, generate_reply_custom

try:
    from .llm_web_search import get_webpage_content, retrieve_from_duckduckgo, retrieve_from_searxng, Generator
    from .retrieval import DocumentRetriever, docs_to_pretty_str
except ImportError:
    from llm_web_search import get_webpage_content, retrieve_from_duckduckgo, retrieve_from_searxng, Generator
    from retrieval import DocumentRetriever, docs_to_pretty_str


params = {
    "display_name": "LLM Web Search",
    "is_tab": True,
    "enable": True,
    "search results per query": 5,
    "langchain similarity score threshold": 0.5,
    "instant answers": True,
    "regular search results": True,
    "search command regex": "",
    "default search command regex": r"Search_web\(\"(.*)\"\)",
    "open url command regex": "",
    "default open url command regex": r"Download_webpage\(\"(.*)\"\)",
    "display search results in chat": True,
    "display extracted URL content in chat": True,
    "searxng url": "",
    "cpu only": True,
    "chunk size": 500,
    "duckduckgo results per query": 10,
    "append current datetime": False,
    "default system prompt filename": None,
    "force search prefix": "Search_web",
    "ensemble weighting": 0.5,
    "keyword retriever": "bm25",
    "splade batch size": 2,
    "chunking method": "character-based",
    "chunker breakpoint_threshold_amount": 30,
    "simple search": False,
    "client timeout": 10,
    "show force search checkbox": True,
    "token classification model id": "mirth/chonky_distilbert_base_uncased_1",
    "think after searching": True
}
custom_system_message_filename = None
extension_path = os.path.dirname(os.path.abspath(__file__))
document_retriever = None
update_history_dict = defaultdict(str)
chat_id = None
force_search = False
GEN_LATENCY_THRESH = 0.002


def setup():
    """
    Is executed when the extension gets imported.
    :return:
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    try:
        with open(os.path.join(extension_path, "settings.json"), "r") as f:
            saved_params = json.load(f)
        params.update(saved_params)
        save_settings()   # add keys of newly added feature to settings.json
    except FileNotFoundError:
        save_settings()

    if not os.path.exists(os.path.join(extension_path, "system_prompts")):
        os.makedirs(os.path.join(extension_path, "system_prompts"))

    toggle_extension(params["enable"])


def save_settings():
    with open(os.path.join(extension_path, "settings.json"), "w") as f:
        json.dump(params, f, indent=4)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return gr.HTML(f'<span style="color:lawngreen"> Settings were saved at {current_datetime}</span>',
                   visible=True)


def toggle_extension(_enable: bool):
    global document_retriever, custom_system_message_filename
    if _enable:
        document_retriever = DocumentRetriever(device="cpu" if params["cpu only"] else "cuda",
                                               keyword_retriever=params["keyword retriever"],
                                               chunking_method=params["chunking method"],
                                               model_cache_dir=os.path.join(extension_path, "hf_models"))
        embedding_model = document_retriever.embedding_model
        embedding_model.to(embedding_model._target_device)
        custom_system_message_filename = params.get("default system prompt filename")
    else:
        if not params["cpu only"]:  # free some VRAM
            model_attrs = ["embedding_model", "splade_doc_model", "splade_query_model"]
            for model_attr in model_attrs:
                if hasattr(document_retriever, model_attr):
                    model = getattr(document_retriever, model_attr)
                    if hasattr(model, "client"):
                        model.client.to("cpu")
                        del model.client
                    else:
                        if hasattr(model, "to"):
                            model.to("cpu")
                        del model
            torch.cuda.empty_cache()
    params.update({"enable": _enable})
    return _enable


def get_available_system_prompts():
    try:
        return ["None"] + sorted(os.listdir(os.path.join(extension_path, "system_prompts")))
    except FileNotFoundError:
        return ["None"]


def load_system_prompt(filename: str or None):
    global custom_system_message_filename
    if not filename:
        return
    if filename == "None" or filename == "Select custom system message to load...":
        custom_system_message_filename = None
        return ""
    with open(os.path.join(extension_path, "system_prompts", filename), "r") as f:
        prompt_str = f.read()

    if params["append current datetime"]:
        prompt_str += f"\nDate and time of conversation: {datetime.now().strftime('%A %d %B %Y %H:%M')}"

    shared.settings['custom_system_message'] = prompt_str
    custom_system_message_filename = filename
    return prompt_str


def save_system_prompt(filename, prompt):
    if not filename:
        return

    with open(os.path.join(extension_path, "system_prompts", filename), "w") as f:
        f.write(prompt)

    return gr.HTML(f'<span style="color:lawngreen"> Saved successfully</span>',
                   visible=True)


def check_file_exists(filename):
    if filename == "":
        return gr.HTML("", visible=False)
    if os.path.exists(os.path.join(extension_path, "system_prompts", filename)):
        return gr.HTML(f'<span style="color:orange"> Warning: Filename already exists</span>', visible=True)
    return gr.HTML("", visible=False)


def timeout_save_message():
    time.sleep(2)
    return gr.HTML("", visible=False)


def deactivate_system_prompt():
    shared.settings['custom_system_message'] = None
    return "None"


def toggle_forced_search(value):
    global force_search
    force_search = value


def update_chat_id(_id):
    global chat_id
    chat_id = _id


def clear_update_history_dict():
    update_history_dict[chat_id] = ""


def get_force_search_box_theme_js():
    with open(f"{extension_path}/force_search_box_theme.js", "r") as f:
        return f.read()


def ui():
    """
    Creates custom gradio elements when the UI is launched.
    :return:
    """
    # Inject custom system message into the main textbox if a default one is set
    shared.gradio['custom_system_message'].value = load_system_prompt(custom_system_message_filename)


    def toggle_full_search_options(simple_search: bool):
        if simple_search:
            return [gr.update(visible=False)] * 7
        else:
            return [gr.update(visible=True)] * 7


    def update_regex_setting(input_str: str, setting_key: str, error_html_element: gr.component):
        if input_str == "":
            params.update({setting_key: params[f"default {setting_key}"]})
            return {error_html_element: gr.HTML("", visible=False)}
        try:
            compiled = re.compile(input_str)
            if compiled.groups > 1:
                raise re.error(f"Only 1 capturing group allowed in regex, but there are {compiled.groups}.")
            params.update({setting_key: input_str})
            return {error_html_element: gr.HTML("", visible=False)}
        except re.error as e:
            return {error_html_element: gr.HTML(f'<span style="color:red"> Invalid regex. {str(e).capitalize()}</span>',
                                                visible=True)}

    def update_default_custom_system_message(check: bool):
        if check:
            params.update({"default system prompt filename": custom_system_message_filename})
        else:
            params.update({"default system prompt filename": None})

    with gr.Row():
        enable = gr.Checkbox(value=lambda: params['enable'], label='Enable LLM web search')
        use_cpu_only = gr.Checkbox(value=lambda: params['cpu only'],
                                   label='Run extension on CPU only '
                                         '(Save settings and restart for the change to take effect)')
        with gr.Column():
            save_settings_btn = gr.Button("Save settings")
            saved_success_elem = gr.HTML("", visible=False)

    with gr.Row():
        with gr.Column():
            search_type = gr.Radio([("Simple search", True), ("Full search", False)], label="Search type",
                               value=lambda: params["simple search"])
        with gr.Column():
            search_command_regex = gr.Textbox(label="Search command regex string",
                                              placeholder=params["default search command regex"],
                                              value=lambda: params["search command regex"])
            search_command_regex_error_label = gr.HTML("", visible=False)

        with gr.Column():
            open_url_command_regex = gr.Textbox(label="Download webpage command regex string",
                                                placeholder=params["default open url command regex"],
                                                value=lambda: params["open url command regex"])
            open_url_command_regex_error_label = gr.HTML("", visible=False)

        with gr.Column():
            show_results = gr.Checkbox(value=lambda: params['display search results in chat'],
                                       label='Display search results in chat')
            show_url_content = gr.Checkbox(value=lambda: params['display extracted URL content in chat'],
                                           label='Display extracted URL content in chat')
            show_force_search = gr.Checkbox(value=lambda: params['show force search checkbox'],
                                           label='Show force search checkbox', elem_id="show-force-search-box")
    gr.Markdown(value='---')
    with gr.Row():
        with gr.Column():
            gr.Markdown(value='#### Load custom system message\n'
                              'Select a saved custom system message from within the system_prompts folder or "None" '
                              'to clear the selection')
            system_prompt = gr.Dropdown(
                choices=get_available_system_prompts(), label="Select custom system message",
                value=lambda: 'Select custom system message to load...' if custom_system_message_filename is None else
                              custom_system_message_filename, elem_classes='slim-dropdown')
            with gr.Row():
                set_system_message_as_default = gr.Checkbox(
                    value=lambda: custom_system_message_filename == params["default system prompt filename"],
                    label='Set this custom system message as the default')
                refresh_button = ui_module.create_refresh_button(system_prompt, lambda: None,
                                                                 lambda: {'choices': get_available_system_prompts()},
                                                                 'refresh-button', interactive=True)
                refresh_button.elem_id = "custom-sysprompt-refresh"
                delete_button = gr.Button('🗑️', elem_classes='refresh-button', interactive=True)
            append_datetime = gr.Checkbox(value=lambda: params['append current datetime'],
                                          label='Append current date and time when loading custom system message')
        with gr.Column():
            gr.Markdown(value='#### Create custom system message')
            system_prompt_text = gr.Textbox(label="Custom system message", lines=3,
                                            value=lambda: load_system_prompt(custom_system_message_filename))
            sys_prompt_filename = gr.Text(label="Filename")
            sys_prompt_save_button = gr.Button("Save Custom system message")
            system_prompt_saved_success_elem = gr.HTML("", visible=False)

    gr.Markdown(value='---')
    with gr.Accordion("Advanced settings", open=False):
        ensemble_weighting = gr.Slider(minimum=0, maximum=1, step=0.05, value=lambda: params["ensemble weighting"],
                                       label="Ensemble Weighting", info="Smaller values = More keyword oriented, "
                                                                        "Larger values = More focus on semantic similarity",
                                       visible=not params["simple search"])
        with gr.Row():
            keyword_retriever = gr.Radio([("Okapi BM25", "bm25"),("SPLADE", "splade")], label="Sparse keyword retriever",
                                         info="For change to take effect, toggle the extension off and on again",
                                         value=lambda: params["keyword retriever"],
                                         visible=not params["simple search"])
            splade_batch_size = gr.Slider(minimum=2, maximum=256, step=2, value=lambda: params["splade batch size"],
                                          label="SPLADE batch size",
                                          info="Smaller values = Slower retrieval (but lower VRAM usage), "
                                               "Larger values = Faster retrieval (but higher VRAM usage). "
                                               "A good trade-off seems to be setting it = 8",
                                          precision=0,
                                          visible=not params["simple search"])
        with gr.Row():
            chunker = gr.Radio([("Character-based", "character-based"),
                                ("Semantic", "semantic"), ("Token Classifier", "token-classifier")],
                               label="Chunking method",
                               value=lambda: params["chunking method"],
                               visible=not params["simple search"])
            chunker_breakpoint_threshold_amount = gr.Slider(minimum=1, maximum=100, step=1,
                                                            value=lambda: params["chunker breakpoint_threshold_amount"],
                                                            label="Semantic chunking: sentence split threshold (%)",
                                                            info="Defines how different two consecutive sentences have"
                                                                 " to be for them to be split into separate chunks",
                                                            precision=0,
                                                            visible=not params["simple search"])
        client_timeout = gr.Number(label="Client timeout (in seconds)", info="When reached, pending or unfinished webpage "
                                         "downloads will be cancelled to start the retrieval process immediately",
                                   minimum=1, maximum=100,
                                   value=lambda: params["client timeout"], precision=0)
        gr.Markdown("**Note: Changing the following might result in DuckDuckGo rate limiting or the LM being overwhelmed**")
        num_search_results = gr.Number(label="Max. search results to return per query", minimum=1, maximum=100,
                                       value=lambda: params["search results per query"], precision=0)
        num_process_search_results = gr.Number(label="Number of search results to process per query", minimum=1,
                                               maximum=100, value=lambda: params["duckduckgo results per query"],
                                               precision=0)
        similarity_score_threshold = gr.Number(label="Similarity Score Threshold", minimum=0., maximum=1.,
                                               value=lambda: params["langchain similarity score threshold"],
                                               info="Discard chunks that are not similar "
                                                    "enough to the search query and hence fall below the threshold.")
        chunk_size = gr.Number(label="Max. chunk size", info="The maximal size of the individual chunks that each webpage will"
                                     " be split into, in characters", minimum=2, maximum=10000,
                               value=lambda: params["chunk size"], precision=0,
                               visible=not params["simple search"])
        think_after_search = gr.Checkbox(label="Enable thinking after searching",
                                         value=lambda: params["think after searching"])

    with gr.Row():
        searxng_url = gr.Textbox(label="SearXNG URL",
                                 value=lambda: params["searxng url"])

    # Event functions to update the parameters in the backend
    enable.input(toggle_extension, enable, enable)
    use_cpu_only.change(lambda x: params.update({"cpu only": x}), use_cpu_only, None)
    save_settings_btn.click(save_settings, None, [saved_success_elem])
    ensemble_weighting.change(lambda x: params.update({"ensemble weighting": x}), ensemble_weighting, None)
    keyword_retriever.change(lambda x: params.update({"keyword retriever": x}), keyword_retriever, None)
    splade_batch_size.change(lambda x: params.update({"splade batch size": x}), splade_batch_size, None)
    chunker.change(lambda x: params.update({"chunking method": x}), chunker, None)
    chunker_breakpoint_threshold_amount.change(lambda x: params.update({"chunker breakpoint_threshold_amount": x}),
                                               chunker_breakpoint_threshold_amount, None)
    client_timeout.change(lambda x: params.update({"client timeout": x}), client_timeout, None)
    num_search_results.change(lambda x: params.update({"search results per query": x}), num_search_results, None)
    num_process_search_results.change(lambda x: params.update({"duckduckgo results per query": x}),
                                      num_process_search_results, None)
    similarity_score_threshold.change(lambda x: params.update({"langchain similarity score threshold": x}),
                                      similarity_score_threshold, None)
    chunk_size.change(lambda x: params.update({"chunk size": x}), chunk_size, None)
    search_type.change(lambda x: params.update({"simple search": x}),
                       search_type,
                       None).then(toggle_full_search_options, search_type, [ensemble_weighting, keyword_retriever,
                                                                            splade_batch_size, chunker, chunk_size,
                                                                            chunker_breakpoint_threshold_amount,
                                                                            client_timeout])

    search_command_regex.change(lambda x: update_regex_setting(x, "search command regex",
                                                               search_command_regex_error_label),
                                search_command_regex, search_command_regex_error_label, show_progress="hidden")

    open_url_command_regex.change(lambda x: update_regex_setting(x, "open url command regex",
                                                                 open_url_command_regex_error_label),
                                  open_url_command_regex, open_url_command_regex_error_label, show_progress="hidden")

    show_results.change(lambda x: params.update({"display search results in chat": x}), show_results, None)
    show_url_content.change(lambda x: params.update({"display extracted URL content in chat": x}), show_url_content,
                            None)
    show_force_search.change(lambda x: params.update({"show force search checkbox": x}), show_force_search, None)
    searxng_url.change(lambda x: params.update({"searxng url": x}), searxng_url, None)

    delete_button.click(
        lambda x: x, system_prompt, gradio('delete_filename')).then(
        lambda: os.path.join(extension_path, "system_prompts", ""), None, gradio('delete_root')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))
    shared.gradio['delete_confirm'].click(
        lambda: "None", None, system_prompt).then(
        None, None, None, _js="() => { document.getElementById('custom-sysprompt-refresh').click() }")
    system_prompt.change(load_system_prompt, system_prompt, shared.gradio['custom_system_message'])
    system_prompt.change(load_system_prompt, system_prompt, system_prompt_text)
    # restore checked state if chosen system prompt matches the default
    system_prompt.change(lambda x: x == params["default system prompt filename"], system_prompt,
                         set_system_message_as_default)
    sys_prompt_filename.change(check_file_exists, sys_prompt_filename, system_prompt_saved_success_elem)
    sys_prompt_save_button.click(save_system_prompt, [sys_prompt_filename, system_prompt_text],
                                 system_prompt_saved_success_elem,
                                 show_progress="hidden").then(timeout_save_message,
                                                              None,
                                                              system_prompt_saved_success_elem,
                                                              _js="() => { document.getElementById('custom-sysprompt-refresh').click() }",
                                                              show_progress="hidden").then(lambda: "", None,
                                                                                        sys_prompt_filename,
                                                                                        show_progress="hidden")
    append_datetime.change(lambda x: params.update({"append current datetime": x}), append_datetime, None)
    # '.input' = only triggers when user changes the value of the component, not a function
    set_system_message_as_default.input(update_default_custom_system_message, set_system_message_as_default, None)

    # A dummy checkbox to enable the actual "Force web search" checkbox to trigger a gradio event
    force_search_checkbox = gr.Checkbox(value=False, visible=False, elem_id="Force-search-checkbox")
    force_search_checkbox.change(toggle_forced_search, force_search_checkbox, None)

    # Add event listener to "Past chats" radio menu to get the current unique chat ID
    shared.gradio['unique_id'].change(update_chat_id, shared.gradio['unique_id'], None)

    # Don't update internal history with search results if last reply was removed
    shared.gradio['Remove last'].click(clear_update_history_dict, None, None)

    # Change font color and invert checkmark color of force search checkbox when dark mode is toggled
    shared.gradio['theme_state'].change(None, None, None, js=f"() => {{ {get_force_search_box_theme_js()}; toggleForceSearchDarkMode() }}")

    think_after_search.change(lambda x: params.update({"think after searching": x}), think_after_search, None)


def get_generation_prompt(state, impersonate=False, enable_thinking=True):
    chat_template_str = state['chat_template_str']
    if state['mode'] != 'instruct':
        chat_template_str = chat.replace_character_names(chat_template_str, state['name1'], state['name2'])

    instruction_template = chat.jinja_env.from_string(state['instruction_template_str'])
    chat_template = chat.jinja_env.from_string(chat_template_str)

    instruct_renderer = partial(
        instruction_template.render,
        builtin_tools=None,
        tools=None,
        tools_in_user_message=False,
        add_generation_prompt=False
    )

    chat_renderer = partial(
        chat_template.render,
        add_generation_prompt=False,
        name1=state['name1'],
        name2=state['name2'],
        user_bio=chat.replace_character_names(state['user_bio'], state['name1'], state['name2']),
    )
    if state["mode"] == "instruct":
        start_turn_str, end_turn_str = chat.get_generation_prompt(instruct_renderer, impersonate, False)
    else:
        start_turn_str, end_turn_str = chat.get_generation_prompt(chat_renderer, impersonate, False)

    if not enable_thinking:
        start_turn_str += chat.get_thinking_suppression_string(instruction_template)

    return start_turn_str, end_turn_str


def custom_generate_reply(question, original_question, state, stopping_strings, is_chat, recursive_call=False):
    """
    Overrides the main text generation function.
    :return:
    """
    if shared.model.__class__.__name__ in ['LlamaServer', 'LlamaCppModel', 'RWKVModel', 'ExllamaModel', 'Exllamav2Model',
                                           'CtransformersModel']:
        generate_func = generate_reply_custom
    else:
        generate_func = generate_reply_HF

    if not params['enable']:
        for reply in generate_func(question, original_question, state, stopping_strings, is_chat=is_chat):
            yield reply
        return

    web_search = False
    read_webpage = False
    max_search_results = int(params["search results per query"])
    instant_answers = params["instant answers"]
    simple_search = params["simple search"]

    document_retriever.num_results = int(params["duckduckgo results per query"])
    document_retriever.similarity_threshold = params["langchain similarity score threshold"]
    document_retriever.chunk_size = params["chunk size"]
    document_retriever.ensemble_weighting = params["ensemble weighting"]
    document_retriever.splade_batch_size = params["splade batch size"]
    document_retriever.chunking_method = params["chunking method"]
    document_retriever.chunker_breakpoint_threshold_amount = params["chunker breakpoint_threshold_amount"]
    document_retriever.client_timeout = params["client timeout"]
    document_retriever.token_classification_model_id = params["token classification model id"]

    search_command_regex = params["search command regex"]
    open_url_command_regex = params["open url command regex"]
    searxng_url = params["searxng url"]
    display_search_results = params["display search results in chat"]
    display_webpage_content = params["display extracted URL content in chat"]

    if search_command_regex == "":
        search_command_regex = params["default search command regex"]
    if open_url_command_regex == "":
        open_url_command_regex = params["default open url command regex"]

    compiled_search_command_regex = re.compile(search_command_regex)
    compiled_open_url_command_regex = re.compile(open_url_command_regex)

    if force_search and not recursive_call:
        question += f" {params['force search prefix']}"

    model_reply_gen = generate_func(question, original_question, state, stopping_strings, is_chat=is_chat)
    reply = None
    for reply in model_reply_gen:

        if force_search and not recursive_call:
            reply = params["force search prefix"] + reply

        search_re_match = compiled_search_command_regex.search(reply)
        if search_re_match is not None:
            yield reply
            model_reply_gen.close()
            original_model_reply = reply
            web_search = True
            search_term = search_re_match.group(1)
            print(f"LLM_Web_search | Searching for {search_term}...")
            reply += "\n```plaintext"
            reply += "\nSearch tool:\n"
            if searxng_url == "":
                search_generator = Generator(retrieve_from_duckduckgo(search_term,
                                                                      document_retriever,
                                                                      max_search_results,
                                                                      simple_search))
            else:
                search_generator = Generator(retrieve_from_searxng(search_term,
                                                                   searxng_url,
                                                                   document_retriever,
                                                                   max_search_results,
                                                                   instant_answers,
                                                                   simple_search))
            try:
                for status_message in search_generator:
                    time.sleep(GEN_LATENCY_THRESH)
                    # Insert zero-width space before '*' to avoid empty line after newline
                    yield original_model_reply + f"\n​*{status_message}*"
                time.sleep(GEN_LATENCY_THRESH)
                yield original_model_reply + "\n​*Is typing...*"
                search_results = docs_to_pretty_str(search_generator.retval)
            except Exception as exc:
                exception_message = str(exc)
                reply += f"The search tool encountered an error: {exception_message}"
                print(f'LLM_Web_search | {search_term} generated an exception: {exception_message}')
            else:
                if search_results != "":
                    reply += search_results
                else:
                    reply += f"\nThe search tool did not return any results."
            reply += "```\n"
            if display_search_results:
                time.sleep(GEN_LATENCY_THRESH)
                yield reply
            break

        open_url_re_match = compiled_open_url_command_regex.search(reply)
        if open_url_re_match is not None:
            yield reply
            model_reply_gen.close()
            original_model_reply = reply
            read_webpage = True
            url = open_url_re_match.group(1)
            print(f"LLM_Web_search | Reading {url}...")
            reply += "\n```plaintext"
            reply += "\nURL opener tool:\n"
            try:
                webpage_content = get_webpage_content(url)
            except Exception as exc:
                reply += f"Couldn't open {url}. Error message: {str(exc)}"
                print(f'LLM_Web_search | {url} generated an exception: {str(exc)}')
            else:
                reply += f"\nText content of {url}:\n"
                reply += webpage_content
            reply += "```\n"
            if display_webpage_content:
                yield reply + "*Is typing...*"
            else:
                yield original_model_reply + "\n*Is typing...*"
            break
        yield reply

    if web_search or read_webpage:
        display_results = web_search and display_search_results or read_webpage and display_webpage_content
        # Add results to context and continue model output
        start_turn_str, end_turn_str = get_generation_prompt(state, enable_thinking=params["think after searching"])
        new_question = question + reply + end_turn_str + start_turn_str
        new_reply = ""
        for new_reply in custom_generate_reply(new_question, new_question, state,
                                               stopping_strings, is_chat=is_chat, recursive_call=True):
            if display_results:
                yield f"{reply}{new_reply}"
            else:
                yield f"{original_model_reply}\n{new_reply}"

        if not display_results:
            update_history_dict[state["unique_id"]] = f"{reply}\n{update_history_dict[state['unique_id']]}"
    else:
        if recursive_call and not display_search_results:
            update_history_dict[state["unique_id"]] = f"{reply}\n{update_history_dict[state['unique_id']]}"


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
    with open(os.path.join(extension_path, "script.js"), "r") as f:
        script_js = f.read()
    force_search_box_theme_js = get_force_search_box_theme_js()
    return script_js + force_search_box_theme_js + ";toggleForceSearchDarkMode();"


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
    if update_history_dict[chat_id]:
        # Replace the last reply in the internal history (which does not contain any search results) with the
        # concatenation of all recursive searches and their model completions, which *do* contain the full results.
        if len(history["internal"]) > 0:
            history["internal"][-1][-1] = update_history_dict[chat_id]
        update_history_dict[chat_id] = ""
    return history
