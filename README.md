# Decomposer

The idea of this project is to train a small language model called Decomposer that takes a **task** as input and **decomposes** it into sub-tasks. Applied recursively, it builds complex solutions from simple pieces.

## Task

By **task** we mean a specification of a function, i.e. what inputs are available and what output is desired.

Inputs can be anything — data, functions, models, APIs — all represented uniformly as documented parameters. The semantics of each parameter and the expected output are described in natural language docstrings. Decomposer must only use built-in Python syntax and the parameters explicitly provided and documented in the function signature — no implicit knowledge of libraries or external APIs.

In Python, a task is a function signature and docstring with no implementation:

```python
# --> Decomposer's input starts here
def travel_advisory(
    cities,
    get_weather,
):
    """Given a list of city names, return a travel advice for each city
    based on its current weather conditions.

    Args:
        cities: list of city names to check
        get_weather: a function that takes a city name and returns current weather conditions
    Returns:
        a dict mapping city names to travel advice strings
    """
# --> Decomposer's input ends here
    raise NotImplementedError
```

The docstring captures everything — what each parameter is, how to use it, what to return. `get_weather` is a tool — a callable provided externally, opaque to Decomposer.

## Decomposition

A **decomposition** of a task is a function body that may define and use **sub-tasks** — inner functions with a `raise NotImplementedError` body, each specified with a signature and docstring. Sub-tasks have the same format as tasks, making them directly and recursively decomposable by the same model.

```python
# --> Decomposer's input starts here
def travel_advisory(
    cities,
    get_weather,
):
    """Given a list of city names, return a travel advice for each city
    based on its current weather conditions.

    Args:
        cities: list of city names to check
        get_weather: a function that takes a city name and returns current weather conditions
    Returns:
        a dict mapping city names to travel advice strings
    """
# --> Decomposer's input ends here
# --> Decomposer's output starts here
    def generate_advice(city, weather):
        """Generate a short travel advice for the given city
        based on its current weather conditions.

        Args:
            city: city name
            weather: current weather conditions for the city
        Returns:
            a travel advice string
        """
        raise NotImplementedError

    advisory = {}
    for city in cities:
        weather = get_weather(city)
        advisory[city] = generate_advice(city, weather)
    return advisory
# --> Decomposer's output ends here
```

`get_weather` is used directly — it is provided. `generate_advice` is a sub-task introduced by Decomposer — its spec is defined here, and it can be decomposed recursively.

### Non-decomposable tasks

Decomposer cannot decompose some kinds of tasks:
- Tasks that ask for specific external knowledge, e.g. "Return the current weather in Tokyo."
- Tasks that require ML models, e.g. "Classify this image." or "Generate text".
- Non-computable tasks, e.g. "Does this program halt?"

We call such tasks **non-decomposable**. Given a non-decomposable task, Decomposer should keep the function body not implemented:
```python
# --> Decomposer's input starts here
def get_weather(city):
    """Return current weather conditions for a given city.

    Args:
        city: city name
    Returns:
        current weather conditions as a string
    """
# --> Decomposer's input ends here
# --> Decomposer's output starts here
    raise NotImplementedError
# --> Decomposer's output ends here
```

If user provides all the necessary tools for solving the task, Decomposer should be able to recursively decompose it to a valid function with all sub-tasks implemented. If at some point Decomposer stucks on a non-decomposable sub-task, it may either try to do backtracking over the recursion tree. If it still stucks, it can ask user to provide solvers of the non-decomposable sub-tasks as input parameters.
