def strip_url_query(url):
    return url.split("?", 1)[0]


def get_trimmed_url(url, max_len):
    trimmed_url = strip_url_query(url)
    if len(trimmed_url) > max_len:
        trimmed_url = trimmed_url[:max_len] + " ..."
    return trimmed_url
