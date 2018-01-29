"""
Common schemas for user input validation
"""

import loudml.errors

from voluptuous import (
    All,
    Any,
    error,
    Invalid,
    Length,
    Match,
    Range,
    Schema,
)

key = All(
   str,
   Length(min=1),
   Match("^[a-zA-Z0-9-_]+$"),
)

def validate(schema, data):
    """
    Validate data against a schema
    """

    try:
        schema(data)
    except Invalid as exn:
        raise loudml.errors.Invalid(exn.error_message, path=exn.path)
