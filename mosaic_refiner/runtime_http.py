"""ComfyUI traffic stays on literal loopback, independently of system proxies."""
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


_OPENER=build_opener(ProxyHandler({}),_NoRedirect())


def urlopen(request,timeout=30):
    url=request.full_url if hasattr(request,'full_url') else request
    try:
        parsed=urlsplit(url)
        allowed=(parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','::1')
                 and parsed.port is not None and not parsed.username and not parsed.password and not parsed.fragment)
    except (TypeError,ValueError):allowed=False
    if not allowed:raise ValueError('ComfyUI通信は選択したloopback APIだけに限定します')
    return _OPENER.open(request,timeout=timeout)
