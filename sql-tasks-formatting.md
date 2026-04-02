# Formatting Text-to-SQL Datasets for Decomposer Training

## Goal

Convert existing text-to-SQL benchmarks (Spider, BIRD) into Decomposer training examples: each example teaches the model to perform one level of task decomposition.

## Source Data

- **Spider** (7000 train, 1034 val): declarative SQL with JOINs, subqueries, UNION/INTERSECT/EXCEPT, GROUP BY, HAVING. No procedural SQL.
- **BIRD** (9428 train): richer SQL with CASE WHEN (734), IIF (162), window functions, nested subqueries. Still no recursive CTEs or loops.
- Both benchmarks ship with SQLite databases (actual tables with data).

## Formatting Approach

### Unified Input Format

Everything the model receives is a single function signature + docstring:
- **Database tables** are function parameters (data) with column descriptions in docstrings
- **Natural language question** is in the docstring, specifying the desired output
- No type annotations — the docstring describes everything
- No domain-specific tools — the model only learns decomposition, not tool usage

### Two Kinds of Training Examples

Each SQL query produces two complementary kinds of training examples:

**1. Decomposition examples** — input has only tables, output defines sub-tasks:
- The model learns to split a complex question into simpler sub-questions
- Output contains inner functions with `raise NotImplementedError`
- No domain-specific tools — fully domain-agnostic skill

**2. Implementation examples** — input has tables + tools, output is executable:
- The model learns to solve a simple sub-task using provided tools
- Tools are documented functions analogous to pandas operations (e.g. `merge`, `filter_in`, `groupby`)
- Output is a function body that calls the provided tools — directly executable for verification

The same model, the same format. Whether Decomposer decomposes or implements depends on what's in the signature — if sufficient tools are provided, it implements; if not, it decomposes further.

### Training Data Generation Pipeline

Given a ground truth SQL query:

1. **Decompose** the *question* (not the SQL) into a tree of sub-tasks. Each sub-task produces an intermediate table. This is done by an LLM, which generates semantically meaningful splits. Multiple decomposition trees per query give data augmentation.

2. **Format each tree node as a training example:**
   - **Internal nodes → decomposition examples:** input has tables only, output defines child sub-tasks and wires them together
   - **Leaf nodes → implementation examples:** input has tables + tools, output calls the tools to produce the result

Each SQL query produces multiple training examples (one per node in the decomposition tree).

### Verification

The decomposition itself contains no executable code — only sub-task specs. To verify correctness:

1. Execute original SQL query on SQLite → ground truth result
2. Ask an LLM to implement each leaf sub-task in pandas (these are simple enough to be reliable)
3. Execute the assembled program → our result
4. Compare results (order-insensitive comparison)

If results match, the decomposition is verified. If the leaf LLM fails on some sub-task, the example is discarded — this is a yield issue, not a correctness issue. We are verifying the **decomposition**, not the leaf implementations.

### Concrete Example

**Question:** "Find the name and location of the stadiums which some concerts happened in the years of both 2014 and 2015."

**Original SQL (Spider, used only for verification):**
```sql
SELECT T2.name, T2.location
FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id
WHERE T1.Year = 2014
INTERSECT
SELECT T2.name, T2.location
FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id
WHERE T1.Year = 2015
```

**Training example (level 0 — root decomposition):**

```python
# --> Decomposer's input starts here
def find_stadiums_with_concerts_both_years(concert, stadium):
    """Find the name and location of stadiums where concerts happened
    in both 2014 and 2015.

    Args:
        concert: table with columns [concert_id, concert_name, theme, stadium_id, year]
        stadium: table with columns [stadium_id, location, name, capacity, highest, lowest, average]
    Returns:
        table with columns [name, location]
    """
# --> Decomposer's input ends here
# --> Decomposer's output starts here
    def concerts_in_target_years(concert):
        """Find all concerts that happened in 2014 or 2015.

        Args:
            concert: table with columns [concert_id, concert_name, theme, stadium_id, year]
        Returns:
            table with same columns, filtered to years 2014 and 2015
        """
        raise NotImplementedError

    def group_by_stadium(concerts, stadium):
        """Join concerts with stadiums and group by stadium (name, location).

        Args:
            concerts: table with columns [concert_id, concert_name, theme, stadium_id, year]
            stadium: table with columns [stadium_id, location, name, capacity, highest, lowest, average]
        Returns:
            grouped object, grouped by [name, location]
        """
        raise NotImplementedError

    def count_distinct_years(grouped_concerts):
        """Count distinct years in each group.

        Args:
            grouped_concerts: grouped object, grouped by [name, location]
        Returns:
            table with columns [name, location, year_count]
        """
        raise NotImplementedError

    def select_stadiums_with_both_years(stadiums):
        """Select only stadiums that had concerts in both years (year_count == 2),
        returning their names and locations.

        Args:
            stadiums: table with columns [name, location, year_count]
        Returns:
            table with columns [name, location]
        """
        raise NotImplementedError

    filtered = concerts_in_target_years(concert)
    grouped = group_by_stadium(filtered, stadium)
    counted = count_distinct_years(grouped)
    return select_stadiums_with_both_years(counted)
# --> Decomposer's output ends here
```

**Training example (level 1 — leaf implementation with tools):**

```python
# --> Decomposer's input starts here
def group_by_stadium(concerts, stadium, merge, groupby):
    """Join concerts with stadiums and group by stadium (name, location).

    Args:
        concerts: table with columns [concert_id, concert_name, theme, stadium_id, year]
        stadium: table with columns [stadium_id, location, name, capacity, highest, lowest, average]
        merge: a function(left, right, on) that inner joins two tables on a key column
        groupby: a function(table, by) that groups a table by columns, returns a grouped object
    Returns:
        grouped object, grouped by [name, location]
    """
# --> Decomposer's input ends here
# --> Decomposer's output starts here
    joined = merge(concerts, stadium, "stadium_id")
    return groupby(joined, ["name", "location"])
# --> Decomposer's output ends here
```

`merge` → `pd.merge`, `groupby` → `DataFrame.groupby`. Verification is straightforward — call the real pandas functions and compare against SQL ground truth.

Not all leaf sub-tasks have clean tool mappings. For example, `concerts_in_target_years` (filter by year) uses pandas operator syntax (`df[df["year"].isin([...])]`) which isn't a single named function. Such leaves are verified via LLM-generated pandas code instead.

### Verification

**Decomposition examples** — an LLM implements each leaf sub-task in pandas, the assembled program runs on the actual SQLite data loaded as DataFrames, and the result is compared against the original SQL output. If the leaf LLM fails, the example is discarded — a yield issue, not a correctness issue.

**Implementation examples** — tools have known pandas implementations, so the output is directly executable. Verification is straightforward: run and compare against SQL ground truth.

## What Decomposer Learns

- How to split a complex task into simpler sub-tasks with precise signatures and natural language specs
- How to implement simple tasks using provided tools
- When to decompose further vs. when to implement directly (depends on available tools)
- The decomposition skill itself is domain-agnostic; only the training data is domain-specific

## Open Questions

- Optimal granularity of decomposition (one clause per node vs. semantically meaningful chunks)
- Whether to include column-level information in table parameter docstrings
- How to handle queries too simple to decompose (single SELECT/WHERE) — skip, or include as leaf-only examples?
- Yield rate: what fraction of valid decompositions survive the LLM leaf implementation + verification step?
