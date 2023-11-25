from duckduckgo_search import DDGS


def dict_list_to_pretty_str(data: list[dict]) -> str:
    ret_str = ""
    if isinstance(data, dict):
        data = [data]
    if isinstance(data, list):
        for i, d in enumerate(data):
            ret_str += f"Result {i+1}\n"
            ret_str += f"Title: {d['title']}\n"
            ret_str += f"{d['body']}\n"
            ret_str += f"Source URL: {d['href']}\n"
        return ret_str
    else:
        raise ValueError("Input must be dict or list[dict]")


def search_duckduckgo(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        answer_list = list(ddgs.answers(query))
        if answer_list:
            answer_dict = answer_list[0]
            answer_dict["title"] = query
            answer_dict["body"] = answer_dict["text"]
            answer_dict["href"] = answer_dict["url"]
            answer_dict.pop('icon', None)
            answer_dict.pop('topic', None)
            answer_dict.pop('text', None)
            answer_dict.pop('url', None)
            return [answer_dict]
        else:
            return list(ddgs.text(query, region='wt-wt', safesearch='moderate', timelimit='y', max_results=max_results))

