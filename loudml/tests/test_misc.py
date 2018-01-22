import datetime
import unittest

from loudml.misc import (
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

        self.assertEqual(to_sec("42"), 42)
        self.assertEqual(to_sec("42s"), 42)
        self.assertEqual(to_sec("42m"), 42 * 60)
        self.assertEqual(to_sec("42h"), 42 * 60 * 60)
        self.assertEqual(to_sec("42d"), 42 * 60 * 60 * 24)
        self.assertEqual(to_sec("42w"), 42 * 60 * 60 * 24 * 7)
        self.assertEqual(to_sec("-42s"), -42)

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
