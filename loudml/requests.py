from loudml import errors
import requests
from urllib.parse import urlencode


DEFAULT_REQUEST_TIMEOUT = 5


def perform_request(
    base_url,
    method,
    url,
    session,
    params=None,
    body=None,
    timeout=None,
    ignore=(),
    headers=None
):
    url = base_url + url
    if params:
        url = '%s?%s' % (url, urlencode(params or {}))

    request = requests.Request(
        method=method, headers=headers, url=url, json=body)
    prepared_request = session.prepare_request(request)
    settings = session.merge_environment_settings(
        prepared_request.url, {}, None, None, None)
    send_kwargs = {'timeout': timeout}
    send_kwargs.update(settings)
    try:
        response = session.send(prepared_request, **send_kwargs)
    except Exception as e:
        if isinstance(e, requests.exceptions.SSLError):
            raise errors.SSLError('N/A', str(e), e)
        if isinstance(e, requests.Timeout):
            raise errors.ConnectionTimeout('TIMEOUT', str(e), e)
        raise errors.ConnectionError('N/A', str(e), e)

    return response


def perform_data_request(
    base_url,
    method,
    url,
    session,
    params=None,
    body=None,
    timeout=None,
    ignore=(),
    headers=None
):
    url = base_url + url
    if params:
        url = '%s?%s' % (url, urlencode(params or {}))

    request = requests.Request(
        method=method, headers=headers, url=url, data=body)
    prepared_request = session.prepare_request(request)
    settings = session.merge_environment_settings(
        prepared_request.url, {}, None, None, None)
    send_kwargs = {'timeout': timeout}
    send_kwargs.update(settings)
    try:
        response = session.send(prepared_request, **send_kwargs)
    except Exception as e:
        if isinstance(e, requests.exceptions.SSLError):
            raise errors.SSLError('N/A', str(e), e)
        if isinstance(e, requests.Timeout):
            raise errors.ConnectionTimeout('TIMEOUT', str(e), e)
        raise errors.ConnectionError('N/A', str(e), e)

    return response
