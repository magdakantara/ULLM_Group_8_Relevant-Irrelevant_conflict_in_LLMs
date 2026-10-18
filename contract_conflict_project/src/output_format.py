"""Syntax-only constraints. This module receives no references or case labels."""
FORMAT_VERSION = 'json-schema-v1'
RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {name: {'type': 'string'} for name in ('answer', 'evidence_A', 'evidence_B')},
    'required': ['answer', 'evidence_A', 'evidence_B'],
    'additionalProperties': False,
}


def prepare_tokenizer(tokenizer):
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    return build_token_enforcer_tokenizer_data(tokenizer)


def new_prefix_constraint(tokenizer_data):
    # Fresh parser state for every example and every retry.
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    return build_transformers_prefix_allowed_tokens_fn(tokenizer_data, JsonSchemaParser(RESPONSE_SCHEMA))
