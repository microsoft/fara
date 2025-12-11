def strip_url_query(url: str) -> str:
    """Remove query parameters from a URL.

    Args:
        url: The URL string to process

    Returns:
        The URL without query parameters

    Example:
        >>> strip_url_query("https://example.com/page?foo=bar")
        'https://example.com/page'
    """
    return url.split("?", 1)[0]


def get_trimmed_url(url: str, max_len: int) -> str:
    """Trim URL to maximum length and add ellipsis if needed.

    Removes query parameters and truncates the URL if it exceeds
    the specified maximum length, appending " ..." to indicate truncation.

    Args:
        url: The URL string to trim
        max_len: Maximum allowed length for the URL

    Returns:
        The trimmed URL string, with " ..." appended if truncated

    Example:
        >>> get_trimmed_url("https://example.com/very/long/path", 20)
        'https://example.com/ ...'
    """
    trimmed_url = strip_url_query(url)
    if len(trimmed_url) > max_len:
        trimmed_url = trimmed_url[:max_len] + " ..."
    return trimmed_url
