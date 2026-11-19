__all__ = ['sample']


def __getattr__(name):
    """Load the image sampler lazily so ``python -m`` does not preload it."""
    if name == "sample":
        from .sample_image import sample
        return sample
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
