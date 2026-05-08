# Decomposer

The idea of this project is to train a small language model called Decomposer that takes a **task** as input and **decomposes** it into sub-tasks. Applied recursively, it builds complex solutions from simple pieces.

## Why Decomposer?

Modern LLMs compute output tokens through attention to the context — a fixed-depth computation per token. This means they cannot loop, cannot maintain exact state, and degrade on complex multi-step reasoning. Yet much of what LLMs are asked to do at inference time is *algorithmic*: filtering, sorting, aggregating, branching, composing API calls. This kind of work is better handled by a standard code interpreter — deterministic, fast, and cheap.

Decomposer separates computation across three substrates:

| Substrate | Responsibility | Properties |
|---|---|---|
| **Decomposer** (SLM) | Translate natural language specs into code | Expensive, stochastic — called as few times as possible |
| **Interpreter** (Python) | Execute the algorithmic skeleton | Deterministic, fast, cheap |
| **Tools** (APIs, DBs, other LLMs) | Interact with the external world | Black-box, defined by their specs |

Decomposer is a *semantic compiler*: it reads function specifications (signatures + docstrings) and produces Python implementations. If the implementation needs helper computations, Decomposer introduces new function specifications for them. The same decomposition procedure can then be applied recursively to those new specifications.

The core setup is static:

1. The input is a function specification.
2. Decomposer outputs a function body.
3. The output may also contain specifications for newly introduced sub-task functions.
4. Each new function specification is another valid input to Decomposer.
5. Recursion stops when a function is implemented directly or left as a non-decomposable primitive.

Available data, tools, models, APIs, and callbacks are represented explicitly as documented function parameters. Decomposer should not rely on implicit global bindings beyond Python built-ins.

## Input-output format

### Input task format

By **task** we mean a *specification of a function* (see example below): what input arguments are available and what output / side effects are desired. Arguments can be anything — data, functions, models, APIs. Input / output types can be annotated both in the function signature and in the docstring. Docstring also describes input / output / side effects semantics in natural language.

```python
# --> Decomposer's input starts here
def find_nearest_pharmacy(
    medicine,
    pharmacies,
    check_availability,
):
    """Find the nearest pharmacy that has the required medicine.

    Args:
        medicine (str): medicine name
        pharmacies (list[str]): list of pharmacies
        check_availability (callable): (pharmacy, medicine) → is the medicine in stock
    Returns:
        str | None: nearest pharmacy with the medicine in stock
    """
# --> Decomposer's input ends here
    raise NotImplementedError
```

### Output decomposition format

A **decomposition** of a task is a function body, optionally accompanied by new function specifications for sub-tasks. Sub-tasks have the same format as tasks, making them directly and recursively decomposable by the same model.

```python
# --> Decomposer's input starts here
def find_nearest_pharmacy(
    medicine,
    pharmacies,
    check_availability,
):
    """Find the nearest pharmacy that has the required medicine.

    Args:
        medicine (str): medicine name
        pharmacies (list[str]): list of pharmacies
        check_availability (callable): (pharmacy, medicine) → is the medicine in stock
    Returns:
        str | None: nearest pharmacy with the medicine in stock
    """
# --> Decomposer's input ends here
# --> Decomposer's output starts here
    pharmacies_with_medicine = [
        p for p in pharmacies
        if check_availability(p, medicine)
    ]
    if not pharmacies_with_medicine:
        return None
    return min(
        pharmacies_with_medicine,
        key=get_distance)
# --> Decomposer's output ends here


# --> New sub-task specs start here
def get_distance(pharmacy):
    """Distance to the pharmacy.

    Args:
        pharmacy (str): pharmacy
    Returns:
        float: distance from current location
    """
    raise NotImplementedError
# --> New sub-task specs end here
```

#### Non-decomposable tasks

A task is **non-decomposable** when Decomposer cannot implement it using Python syntax and the documented function parameters without introducing a sub-task that is essentially the same problem. Given a non-decomposable task, Decomposer keeps the function body not implemented:

```python
# --> Decomposer's input starts here
def get_distance(pharmacy):
    """Distance to the pharmacy from current location.

    Args:
        pharmacy (str): pharmacy
    Returns:
        float: distance from current location
    """
# --> Decomposer's input ends here
# --> Decomposer's output starts here
    raise NotImplementedError
# --> Decomposer's output ends here
```

Common reasons a task may be non-decomposable:
- It requires external knowledge or tools, e.g. "Return the current weather in Tokyo."
- It requires ML models, e.g. "Classify this image."
- It is already a primitive operation relative to the current function specification and available inputs.

Both decomposable and non-decomposable sub-tasks share the same syntax (`raise NotImplementedError`). This uniformity is a deliberate design choice — Decomposer learns *when and how to split*, not *what kind of thing the sub-task is*.

### Language-agnostic formatting

The general idea is language-agnostic — Decomposer can be trained simultaneously to work in many languages. Here is the example of the same task decomposition formatted in Haskell:

```haskell
-- --> Decomposer's input starts here
import Data.List (minimumBy)
import Data.Ord (comparing)

-- | Find the nearest pharmacy
--   that has the required medicine.
findNearestPharmacy
    :: String
    -- ^ medicine name
    -> [String]
    -- ^ list of pharmacies
    -> (String       -- pharmacy
        -> String    -- medicine
        -> Bool)
    -- ^ is the medicine in stock
    -> Maybe String
    -- ^ nearest pharmacy in stock
-- --> Decomposer's input ends here
-- --> Decomposer's output starts here

findNearestPharmacy medicine pharmacies
    checkAvailability =
  case pharmaciesWithMedicine of
    [] -> Nothing
    _  -> Just (minimumBy
           (comparing getDistance)
           pharmaciesWithMedicine)
  where
    pharmaciesWithMedicine =
      [ p | p <- pharmacies
          , checkAvailability p medicine ]

    -- | Distance to the pharmacy
    --   from current location.
    getDistance :: String -> Float
    getDistance = undefined
-- --> Decomposer's output ends here
```

A non-decomposable task in Haskell:

```haskell
-- --> Decomposer's input starts here

-- | Distance to the pharmacy
--   from current location.
getDistance :: String -> Float
-- --> Decomposer's input ends here
-- --> Decomposer's output starts here
getDistance = undefined
-- --> Decomposer's output ends here
```
