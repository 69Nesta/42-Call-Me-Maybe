class CallMeMaybeError(ValueError):
    """Base exception for Call Me Maybe.

    All project-specific exceptions inherit from this class so callers can
    catch a single common base type.
    """
    pass
