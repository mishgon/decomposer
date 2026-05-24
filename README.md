# Decomposer

The idea of this project is to train a small language model called Decomposer that takes a **task** as input and **decomposes** it into sub-tasks. Applied recursively, it builds complex solutions from simple pieces.

## Input-output format

### Input task format

We represent a **task** as a *specification of a function* (see example below): what input arguments are available and what output / side effects are desired. Arguments can be anything — data, functions, models, APIs. Input / output types can be annotated in the docstring. Docstring also describes input / output / side effects semantics in natural language.

```python
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
```

### Output decomposition format

By **decomposition** of a task we mean a function body, optionally accompanied by new function specifications for sub-tasks. Sub-tasks have the same format as tasks, making them directly and recursively decomposable by the same model.

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


def get_distance(pharmacy):
    """Distance to the pharmacy.

    Args:
        pharmacy (str): pharmacy
    Returns:
        float: distance from current location
    """
    raise NotImplementedError
# --> Decomposer's output ends here
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

Reasons a task may be non-decomposable:
- It requires external knowledge, e.g. "Return the current weather in Tokyo."
- It requires ML models, e.g. "Classify this image."
- It is a truly non-computable task, e.g. "Does this program halt?"

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
