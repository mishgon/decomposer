from symtrace.benchmark import HandleEnv, symbolic_equal


def test_handle_env_executes_tool_calls():
    env = HandleEnv("(x + x**2)/(x*sin(y)**2 + x*cos(y)**2)")
    first = env.call("sympy.trigsimp", expr="$0")
    second = env.call("sympy.cancel", expr=first["result"])
    assert second["value"]["display"] == "x + 1"


def test_handle_env_executes_branch_tools():
    env = HandleEnv("sin(x) - 1")
    first = env.call("sympy.inspect_branches", expr="$0", symbol_names=["x"])
    assert "branch_id" in first["value"]["display"]
    second = env.call("sympy.solve_branch", expr="$0", symbol_names=["x"], branch_id="b0")
    assert "branch_id" in second["value"]["display"]
    assert "resolved_value" in second["value"]["display"]


def test_symbolic_equal():
    assert symbolic_equal("x + 1", "1 + x")
