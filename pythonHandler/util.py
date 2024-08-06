from urllib.parse import urlparse

def is_valid_https_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme == 'https', result.netloc, result.path])
    except:
        return False
    
