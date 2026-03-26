from sympy import simplify

from symtrace.index_repo import build_function_catalog


def test_build_function_catalog_for_simplify():
    catalog = build_function_catalog([simplify])
    assert len(catalog) == 1
    assert catalog[0].func_id.endswith(".simplify")
    assert catalog[0].ast_hash
