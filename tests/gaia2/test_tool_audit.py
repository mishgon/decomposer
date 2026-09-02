from gyms.gaia2.audit_tools import run_audit


def test_complete_registry_and_representative_surfaces_are_lossless():
    report = run_audit()

    assert report["issues"] == {
        "broad_object_parameters": [],
        "defaulted_required": [],
        "duplicate_tool_names": [],
        "exposed_variadics": [],
        "hidden_tool_leaks": [],
        "invalid_json_schemas": [],
        "invalid_type_names": [],
        "invalid_retry_schemas": [],
        "langchain_broker_differences": [],
        "missing_decomposer_retry_handlers": [],
        "missing_representative_data": [],
        "native_broker_differences": [],
        "native_fallback_or_conversion": [],
        "ordering_differences": [],
        "retry_hidden_parameter_leaks": [],
        "retry_schema_differences": [],
        "simple_decomposer_retry_differences": [],
        "surface_differences": [],
        "unsupported_annotations": [],
    }
    assert report["ok"] is True
    assert report["registry"] == {
        "app_count": 20,
        "tool_count": 161,
        "python_argument_count": 347,
        "public_argument_count": 332,
    }
    for domain in ("execution", "search", "ambiguity"):
        surface = report["representative_scenarios"][domain]
        assert surface["registered_tools"] == 103
        assert surface["simple_tools"] == 99
        assert surface["decomposer_tools"] == 98
        assert surface["common_tools"] == 98
        assert surface["simple_only_tools"] == [
            "AgentUserInterface__send_message_to_user"
        ]
