import datetime
import unittest

from loudml import (
    errors,
)

from loudml.misc import (
    deepsizeof,
    escape_quotes,
    escape_doublequotes,
    make_datetime,
    make_ts,
    str_to_ts,
    ts_to_str,
    parse_addr,
    parse_timedelta,
)

class TestMisc(unittest.TestCase):
    def test_timedelta(self):
        def to_sec(string):
            return parse_timedelta(string).total_seconds()

        self.assertEqual(to_sec("4"), 4)
        self.assertEqual(to_sec("42"), 42)
        self.assertEqual(to_sec("+42"), 42)
        self.assertEqual(to_sec("42s"), 42)
        self.assertEqual(to_sec("42.0s"), 42.0)
        self.assertEqual(to_sec("42m"), 42 * 60)
        self.assertEqual(to_sec("42h"), 42 * 60 * 60)
        self.assertEqual(to_sec("42d"), 42 * 60 * 60 * 24)
        self.assertEqual(to_sec("42w"), 42 * 60 * 60 * 24 * 7)
        self.assertEqual(to_sec("-42s"), -42)
        self.assertEqual(to_sec("2M"), 60 * 24 * 3600)
        self.assertEqual(to_sec("2y"), 365 * 2 * 24 * 3600)

        def invalid(value, **kwargs):
            with self.assertRaises(errors.Invalid):
                parse_timedelta(value, **kwargs)

        invalid("")
        invalid("foo")
        invalid("42x")
        invalid(-42, min=0)
        invalid("0w", min=0, min_included=False)
        invalid(43, max=42)
        invalid(42, max=42, max_included=False)

    def test_datetime(self):
        expected = datetime.datetime(
            year=2018,
            month=1,
            day=8,
            hour=9,
            minute=39,
            second=26,
            microsecond=123000,
            tzinfo=datetime.timezone.utc,
        )
        self.assertEqual(
            make_datetime(1515404366.123),
            expected,
        )
        self.assertEqual(
            make_datetime("2018-01-08T09:39:26.123Z"),
            expected,
        )
        self.assertEqual(
            make_ts(1515404366.123),
            1515404366.123,
        )
        self.assertEqual(
            make_ts("2018-01-08T09:39:26.123Z"),
            1515404366.123,
        )
        self.assertEqual(
            ts_to_str(1515404366.123),
            "2018-01-08T09:39:26.123Z",
        )
        self.assertEqual(
            str_to_ts("2018-01-08T09:39:26.123Z"),
            1515404366.123,
        )

        with self.assertRaises(ValueError):
            make_datetime(253536624000.0)

    def test_parse_addr(self):
        self.assertEqual(
            parse_addr("localhost", default_port=80),
            {
                'host': "localhost",
                'port': 80,
            }
        )

        self.assertEqual(
            parse_addr("localhost:8080", default_port=80),
            {
                'host': "localhost",
                'port': 8080,
            }
        )

    def test_deepsizeof(self):
        size = deepsizeof({
            'i': 0,
            'f': 1.0,
            'a': [1.0, 2.0, 3.0],
        })

        # XXX It seems that loading some python modules may change the result
        self.assertTrue(622 <= size <= 646)

    def test_escape_quotes(self):
        self.assertEqual(escape_quotes("foo ' '"), "foo \\' \\'")

    def test_escape_doublequotes(self):
        self.assertEqual(escape_doublequotes('foo " "'), 'foo \\" \\"')
