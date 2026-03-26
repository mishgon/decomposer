# Chain Examples

## simplify_mixed
- example_family: `simplify_mixed`
- episode_id: `simplify_18`
- entry_function: `simplify`
- pattern_label: `mixed`
- input: `((x**2 - 1)/(x - 1))*(sin(y)**2 + cos(y)**2)`
- final_output: `x + 1`
- raw_num_calls: `14`

### Raw Call Prefix

- depth 0: `sympy.simplify.simplify.simplify` -> `x + 1`
- depth 1: `sympy.simplify.simplify.simplify` -> `y`
- depth 1: `sympy.simplify.simplify.simplify` -> `y`
- depth 1: `sympy.simplify.powsimp.powsimp` -> `(x**2 - 1)*(sin(y)**2 + cos(y)**2)/(x - 1)`
- depth 1: `sympy.polys.polytools.cancel` -> `x*sin(y)**2 + x*cos(y)**2 + sin(y)**2 + cos(y)**2`
- depth 1: `sympy.polys.rationaltools.together` -> `(x**2 - 1)*(sin(y)**2 + cos(y)**2)/(x - 1)`
- depth 1: `sympy.polys.rationaltools.together` -> `x*sin(y)**2 + x*cos(y)**2 + sin(y)**2 + cos(y)**2`
- depth 1: `sympy.core.exprtools.factor_terms` -> `(x**2 - 1)*(sin(y)**2 + cos(y)**2)/(x - 1)`
- depth 1: `sympy.simplify.hyperexpand.hyperexpand` -> `(x**2 - 1)*(sin(y)**2 + cos(y)**2)/(x - 1)`
- depth 1: `sympy.simplify.trigsimp.trigsimp` -> `(x**2 - 1)/(x - 1)`
- depth 1: `sympy.simplify.powsimp.powsimp` -> `(x**2 - 1)/(x - 1)`
- depth 1: `sympy.simplify.powsimp.powsimp` -> `(x**2 - 1)/(x - 1)`

### Abstract Tool Steps

- `sympy.rational_simplify` with args `{'expr': '$0'}` -> `(x**2 - 1)*(sin(y)**2 + cos(y)**2)/(x - 1)`
- `sympy.trig_simplify` with args `{'expr': '$1'}` -> `(x**2 - 1)/(x - 1)`
- `sympy.rational_simplify` with args `{'expr': '$2'}` -> `x + 1`

## simplify_mixed
- example_family: `simplify_mixed`
- episode_id: `simplify_3`
- entry_function: `simplify`
- pattern_label: `mixed`
- input: `((sin(x)**2 + cos(x)**2) + (sin(x)**2 + cos(x)**2))`
- final_output: `2`
- raw_num_calls: `14`

### Raw Call Prefix

- depth 0: `sympy.simplify.simplify.simplify` -> `2`
- depth 1: `sympy.simplify.simplify.simplify` -> `x`
- depth 1: `sympy.simplify.simplify.simplify` -> `x`
- depth 1: `sympy.simplify.powsimp.powsimp` -> `2*sin(x)**2 + 2*cos(x)**2`
- depth 1: `sympy.polys.polytools.cancel` -> `2*sin(x)**2 + 2*cos(x)**2`
- depth 1: `sympy.polys.rationaltools.together` -> `2*(sin(x)**2 + cos(x)**2)`
- depth 1: `sympy.polys.rationaltools.together` -> `2*(sin(x)**2 + cos(x)**2)`
- depth 1: `sympy.core.exprtools.factor_terms` -> `2*(sin(x)**2 + cos(x)**2)`
- depth 1: `sympy.simplify.hyperexpand.hyperexpand` -> `2*(sin(x)**2 + cos(x)**2)`
- depth 1: `sympy.simplify.trigsimp.trigsimp` -> `2`
- depth 1: `sympy.simplify.powsimp.powsimp` -> `2`
- depth 1: `sympy.simplify.powsimp.powsimp` -> `2`

### Abstract Tool Steps

- `sympy.rational_simplify` with args `{'expr': '$0'}` -> `2*(sin(x)**2 + cos(x)**2)`
- `sympy.trig_simplify` with args `{'expr': '$1'}` -> `2`

## solve_branching
- example_family: `solve_branching`
- episode_id: `solve_3`
- entry_function: `solve`
- pattern_label: `mixed`
- input: `sin(x) - 1`
- final_output: `[pi/2]`
- raw_num_calls: `9`

### Raw Call Prefix

- depth 0: `sympy.solvers.solvers.solve` -> `[pi/2]`
- depth 1: `sympy.solvers.solvers._solve` -> `[{x: pi/2}]`
- depth 2: `sympy.solvers.solvers.solve_linear` -> `(sin(x) - 1, 1)`
- depth 2: `sympy.solvers.solvers._solve` -> `[{x: asin(_t)}, {x: pi - asin(_t)}]`
- depth 3: `sympy.solvers.solvers.solve_linear` -> `(-_t + sin(x), 1)`
- depth 3: `sympy.solvers.solvers._solve` -> `[{x: asin(_t)}]`
- depth 4: `sympy.solvers.solvers.solve_linear` -> `(x, asin(_t))`
- depth 3: `sympy.solvers.solvers._solve` -> `[{x: pi - asin(_t)}]`
- depth 4: `sympy.solvers.solvers.solve_linear` -> `(x, pi - asin(_t))`

### Abstract Tool Steps

- `sympy.solve` with args `{'expr': '$0', 'symbol_names': ['x']}` -> `[pi/2]`
- `sympy.inspect_branches` with args `{'expr': '$1', 'symbol_names': ['x']}` -> `[{'branch_id': 'b0', 'binding': '{x: pi/2}', 'resolved_value': '{x: pi/2}'}, {'branch_id': 'b1', 'binding': '{x: asin(_t)}', 'resolved_value': '{x: asin(_t)}'}, {'branch_id': 'b2', 'binding': '{x: pi - asin(_t)}', 'resolved_value': '{x: pi - asin(_t)}'}]`
- `sympy.solve_branch` with args `{'expr': '$2', 'symbol_names': ['x'], 'branch_id': 'b0'}` -> `{'branch_id': 'b0', 'binding': '{x: pi/2}', 'resolved_value': '{x: pi/2}'}`
- `sympy.solve_branch` with args `{'expr': '$3', 'symbol_names': ['x'], 'branch_id': 'b1'}` -> `{'branch_id': 'b1', 'binding': '{x: asin(_t)}', 'resolved_value': '{x: asin(_t)}'}`
- `sympy.solve_branch` with args `{'expr': '$4', 'symbol_names': ['x'], 'branch_id': 'b2'}` -> `{'branch_id': 'b2', 'binding': '{x: pi - asin(_t)}', 'resolved_value': '{x: pi - asin(_t)}'}`

## solve_branching
- example_family: `solve_branching`
- episode_id: `solve_0`
- entry_function: `solve`
- pattern_label: `chain`
- input: `x**2 - 1`
- final_output: `[-1, 1]`
- raw_num_calls: `3`

### Raw Call Prefix

- depth 0: `sympy.solvers.solvers.solve` -> `[-1, 1]`
- depth 1: `sympy.solvers.solvers._solve` -> `[{x: -1}, {x: 1}]`
- depth 2: `sympy.solvers.solvers.solve_linear` -> `(x**2 - 1, 1)`

### Abstract Tool Steps

- `sympy.solve` with args `{'expr': '$0', 'symbol_names': ['x']}` -> `[-1, 1]`
- `sympy.inspect_branches` with args `{'expr': '$1', 'symbol_names': ['x']}` -> `[{'branch_id': 'b0', 'binding': '{x: -1}', 'resolved_value': '{x: -1}'}, {'branch_id': 'b1', 'binding': '{x: 1}', 'resolved_value': '{x: 1}'}]`
- `sympy.solve_branch` with args `{'expr': '$2', 'symbol_names': ['x'], 'branch_id': 'b0'}` -> `{'branch_id': 'b0', 'binding': '{x: -1}', 'resolved_value': '{x: -1}'}`
- `sympy.solve_branch` with args `{'expr': '$3', 'symbol_names': ['x'], 'branch_id': 'b1'}` -> `{'branch_id': 'b1', 'binding': '{x: 1}', 'resolved_value': '{x: 1}'}`

## factor_decompose
- example_family: `factor_decompose`
- episode_id: `factor_0`
- entry_function: `factor`
- pattern_label: `chain`
- input: `(((x + x) + x) + x)`
- final_output: `4*x`
- raw_num_calls: `3`

### Raw Call Prefix

- depth 0: `sympy.polys.polytools.factor` -> `4*x`
- depth 1: `sympy.polys.polytools._generic_factor` -> `4*x`
- depth 2: `sympy.polys.polytools._symbolic_factor` -> `4*x`

### Abstract Tool Steps

- `sympy.factor` with args `{'expr': '$0'}` -> `4*x`
- `sympy.factor_decompose` with args `{'expr': '$1'}` -> `4*x`

## factor_decompose
- example_family: `factor_decompose`
- episode_id: `factor_1`
- entry_function: `factor`
- pattern_label: `chain`
- input: `sin((((y + 1) + 1) + 1))`
- final_output: `sin(y + 3)`
- raw_num_calls: `3`

### Raw Call Prefix

- depth 0: `sympy.polys.polytools.factor` -> `sin(y + 3)`
- depth 1: `sympy.polys.polytools._generic_factor` -> `sin(y + 3)`
- depth 2: `sympy.polys.polytools._symbolic_factor` -> `sin(y + 3)`

### Abstract Tool Steps

- `sympy.factor` with args `{'expr': '$0'}` -> `sin(y + 3)`
- `sympy.factor_decompose` with args `{'expr': '$1'}` -> `sin(y + 3)`
