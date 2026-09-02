from gyms.gaia2.audit_tools import run_audit


def test_complete_registry_and_representative_surfaces_are_lossless():
    report = run_audit()

    assert report["issues"] == {
        "broad_object_parameters": [],
        "defaulted_required": [],
        "duplicate_tool_names": [],
        "empty_string_payload_optional": [],
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
        # Aliased apps remain separate because every public name is a distinct
        # schema surface shown to the model; do not deduplicate these by method.
        "empty_string_payload_arguments": [
            ["AgentUserInterface__send_message_to_user", "content"],
            ["EmailClientV2__reply_to_email", "content"],
            ["EmailClientV2__send_email", "content"],
            ["EmailClientV2__send_email", "subject"],
            ["Mail__reply_to_email", "content"],
            ["Mail__send_email", "content"],
            ["Mail__send_email", "subject"],
            ["MessagingAppV2__send_message", "content"],
            ["MessagingAppV2__send_message_to_group_conversation", "content"],
            ["ShoppingApp__get_discount_code_info", "discount_code"],
            ["Shopping__get_discount_code_info", "discount_code"],
        ],
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
