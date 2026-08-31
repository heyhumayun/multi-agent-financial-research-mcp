from __future__ import annotations

import ssl


def trusted_ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle when the local Python install lacks one."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()
