__version__ = "unknown"
try:
    from ._version import __version__
    print(__version__)
except ImportError:
    pass
