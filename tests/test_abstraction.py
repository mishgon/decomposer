from symtrace.abstraction import abstract_episode


def test_abstract_episode_drops_repeated_unchanged_steps():
    raw_episode = {
        "episode": {"call_ids": ["c1", "c2", "c3"]},
        "calls": {
            "c1": {
                "func_id": "sympy.polys.polytools.cancel",
                "inputs": {"arg0": {"display": "x + 1"}},
                "output": {"display": "x + 1"},
            },
            "c2": {
                "func_id": "sympy.polys.polytools.cancel",
                "inputs": {"arg0": {"display": "x + 1"}},
                "output": {"display": "x + 1"},
            },
            "c3": {
                "func_id": "sympy.simplify.trigsimp.trigsimp",
                "inputs": {"arg0": {"display": "sin(x)**2 + cos(x)**2"}},
                "output": {"display": "1"},
            },
        },
    }
    steps = abstract_episode(raw_episode)
    assert len(steps) == 1
    assert steps[0]["tool"] == "sympy.trig_simplify"


def test_abstract_episode_falls_back_to_root_simplify_when_helpers_are_empty():
    raw_episode = {
        "episode": {"call_ids": ["c1"], "root_call_id": "c1"},
        "calls": {
            "c1": {
                "func_id": "sympy.simplify.simplify.simplify",
                "inputs": {"arg0": {"display": "x + x"}},
                "output": {"display": "2*x"},
            },
        },
    }
    steps = abstract_episode(raw_episode)
    assert len(steps) == 1
    assert steps[0]["tool"] == "sympy.simplify"


def test_abstract_episode_compresses_rational_cluster_into_macro_step():
    raw_episode = {
        "episode": {"call_ids": ["c1", "c2", "c3"], "root_call_id": "c1"},
        "calls": {
            "c1": {
                "func_id": "sympy.simplify.simplify.simplify",
                "inputs": {"arg0": {"display": "(x**2 + x)/(x*sin(y)**2 + x*cos(y)**2)"}},
                "output": {"display": "x + 1"},
            },
            "c2": {
                "func_id": "sympy.polys.rationaltools.together",
                "inputs": {"arg0": {"display": "(x**2 + x)/(x*sin(y)**2 + x*cos(y)**2)"}},
                "output": {"display": "(x + 1)/(sin(y)**2 + cos(y)**2)"},
            },
            "c3": {
                "func_id": "sympy.polys.polytools.cancel",
                "inputs": {"arg0": {"display": "(x + 1)/(sin(y)**2 + cos(y)**2)"}},
                "output": {"display": "x + 1"},
            },
        },
    }
    steps = abstract_episode(raw_episode)
    assert [step["tool"] for step in steps] == ["sympy.rational_simplify"]


def test_abstract_episode_expands_solve_cluster_into_branch_steps():
    raw_episode = {
        "episode": {"call_ids": ["c1", "c2", "c3"], "root_call_id": "c0"},
        "calls": {
            "c1": {
                "func_id": "sympy.solvers.solvers._solve",
                "inputs": {"arg0": {"display": "sin(x) - 1"}, "arg1": {"display": "x"}},
                "output": {"display": "[{x: asin(_t)}, {x: pi - asin(_t)}]"},
            },
            "c2": {
                "func_id": "sympy.solvers.solvers.solve_linear",
                "inputs": {"arg0": {"display": "-_t + sin(x)"}, "arg1": {"display": "1"}},
                "output": {"display": "(-_t + sin(x), 1)"},
            },
            "c3": {
                "func_id": "sympy.solvers.solvers._solve",
                "inputs": {"arg0": {"display": "x"}, "arg1": {"display": "asin(_t)"}},
                "output": {"display": "[{x: asin(_t)}]"},
            },
        },
    }
    steps = abstract_episode(raw_episode)
    assert steps[0]["tool"] == "sympy.inspect_branches"
    assert steps[1]["tool"] == "sympy.solve_branch"
    assert steps[2]["tool"] == "sympy.solve_branch"
