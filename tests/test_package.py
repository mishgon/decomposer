from decomposer import create_decomposer_agent


def test_package_imports() -> None:
    assert callable(create_decomposer_agent)
