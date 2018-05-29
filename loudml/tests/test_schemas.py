import unittest

import loudml.vendor
from loudml import (
    errors,
    schemas,
)

class TestSchemas(unittest.TestCase):
    def valid(self, value):
        schemas.validate(self.schema, value)

    def invalid(self, value):
        with self.assertRaises(errors.Invalid):
            schemas.validate(self.schema, value)

    def test_key(self):
        self.schema = schemas.key

        self.valid("foo")
        self.valid("foo_bar")
        self.valid("Foo-Bar")
        self.valid("00_foo_00_bar_001")
        self.valid("_foo")

        self.invalid("")
        self.invalid("foo/bar")
        self.invalid(".foo")

    def test_timestamp(self):
        self.schema = schemas.Timestamp()

        self.valid("now")
        self.valid("now-1d")
        self.valid("2018-01-08T09:39:26.123Z")
        self.valid("1515404366.123")
        self.valid(1515404366.123)

        self.invalid("")
        self.invalid(None)
        self.invalid("foo")
