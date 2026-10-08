from plskit import _api

import plskit


def test_every_public_function_converts_engine_errors():
    # Every `_convert_errors` wrapper shares one code object. A raw PyO3
    # builtin has no `__code__`, and a `functools.wraps` wrapper from any
    # other decorator has a different one, so both are reported.
    convert_code = _api._convert_errors(lambda: None).__code__
    functions = [
        (name, obj)
        for name, obj in ((name, getattr(plskit, name)) for name in plskit.__all__)
        if callable(obj) and not isinstance(obj, type)
    ]
    assert functions
    unconverted = [
        name for name, f in functions if getattr(f, "__code__", None) is not convert_code
    ]
    assert unconverted == []
