from eval import filter_examples_by_num_hops


def test_filter_examples_by_num_hops_keeps_matching_examples():
    examples = [
        {"id": "2hop", "question_decomposition": [{}, {}]},
        {"id": "3hop", "question_decomposition": [{}, {}, {}]},
        {"id": "4hop", "question_decomposition": [{}, {}, {}, {}]},
    ]

    assert filter_examples_by_num_hops(examples, 3) == [examples[1]]


def test_filter_examples_by_num_hops_returns_all_examples_without_filter():
    examples = [
        {"id": "2hop", "question_decomposition": [{}, {}]},
        {"id": "3hop", "question_decomposition": [{}, {}, {}]},
    ]

    assert filter_examples_by_num_hops(examples, None) == examples
