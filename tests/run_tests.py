#!/usr/bin/env python3
"""UCEF test runner — works without pytest. Uses built-in unittest only."""
import sys
import os
import traceback
import types

# Ensure project root and src are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, 'src'))
sys.path.insert(0, _project_root)

# Minimal pytest shim so test files can import pytest
if 'pytest' not in sys.modules:
    _pytest = types.ModuleType('pytest')

    def _approx(expected, rel=None, abs=None):
        class Approx:
            def __init__(self, v, rel, ab):
                self.v = v
                self.rel = rel
                self.ab = ab or 1e-6
            def __eq__(self, other):
                return abs(self.v - other) <= self.ab
            def __repr__(self):
                return f'approx({self.v})'
        return Approx(expected, rel, abs)

    def _raises(exc_type, match=None):
        class RaisesCtx:
            def __enter__(self):
                return self
            def __exit__(self, etype, eval, tb):
                if etype is None:
                    raise AssertionError(f'Expected {exc_type.__name__} was not raised')
                if not issubclass(etype, exc_type):
                    return False
                return True
        return RaisesCtx()

    _pytest.approx = _approx
    _pytest.raises = _raises
    sys.modules['pytest'] = _pytest


import importlib

test_modules = [
    'tests.test_types',
    'tests.test_config',
    'tests.test_retrieval',
    'tests.test_memory',
    'tests.test_compression',
    'tests.test_quality_models',
    'tests.test_physics',
    'tests.test_system_e2e',
]

total = passed = failed = errors = 0
failures = []

for mod_name in test_modules:
    print(f'\n=== {mod_name} ===')
    try:
        mod = importlib.import_module(mod_name)
        test_classes = [v for k, v in vars(mod).items()
                        if isinstance(v, type) and k.startswith('Test')]
        for cls in test_classes:
            instance = cls()
            test_methods = sorted([m for m in dir(instance) if m.startswith('test_')])
            for method_name in test_methods:
                total += 1
                try:
                    if hasattr(instance, 'setup_method'):
                        instance.setup_method()
                    getattr(instance, method_name)()
                    passed += 1
                    print(f'  PASS  {cls.__name__}.{method_name}')
                except AssertionError as e:
                    failed += 1
                    msg = str(e)[:100]
                    failures.append(f'{cls.__name__}.{method_name}: {msg}')
                    print(f'  FAIL  {cls.__name__}.{method_name}: {msg}')
                except Exception as e:
                    errors += 1
                    msg = f'{type(e).__name__}: {str(e)[:100]}'
                    failures.append(f'{cls.__name__}.{method_name}: {msg}')
                    print(f'  ERROR {cls.__name__}.{method_name}: {msg}')
    except Exception as e:
        print(f'  IMPORT ERROR: {type(e).__name__}: {str(e)[:150]}')

print(f'\n{"="*60}')
print(f'Total: {total}  Passed: {passed}  Failed: {failed}  Errors: {errors}')
if failures:
    print(f'\nFailures/Errors ({len(failures)}):')
    for f in failures:
        print(f'  - {f}')

result = 'ALL PASSED' if failed == 0 and errors == 0 else f'{failed+errors} ISSUE(S)'
print(f'\nResult: {result}')
sys.exit(0 if failed == 0 and errors == 0 else 1)
