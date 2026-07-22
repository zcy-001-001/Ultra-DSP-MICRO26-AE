import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_ooc_reports.py"
TABLE3_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "table3_figure18"
    / "evidence"
    / "table3"
)
SPEC = importlib.util.spec_from_file_location("extract_ooc_reports", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ParserTests(unittest.TestCase):
    def test_utilization_parser_uses_top_level_rows(self):
        text = """
| CLB LUTs | 244939 | 0 | 0 | 1303680 | 18.79 |
| CLB Registers | 259987 | 0 | 0 | 2607360 | 9.97 |
| DSPs | 4096 | 0 | 0 | 9024 | 45.39 |
| CLB LUTs | 12 | 13 | 14 | 0 | 0 |
"""
        self.assertEqual(
            MODULE.parse_utilization(text),
            {"lut": 244939, "ff": 259987, "dsp": 4096},
        )

    def test_power_parser(self):
        text = "| Total On-Chip Power (W) | 6.218 |"
        self.assertEqual(MODULE.parse_power(text), 6.218)

    def test_wns_parser(self):
        text = """WNS(ns) TNS(ns) endpoints
------- ------- ---------
  0.840 0.000 0
"""
        self.assertEqual(MODULE.parse_wns(text), 0.840)

    def test_absolute_report_path_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE._validate_relative("/private/report.rpt")

    def test_packaged_table3_evidence(self):
        rows = MODULE.extract_table3(MODULE.ReportReader(str(TABLE3_EVIDENCE)))
        MODULE.validate_table3(rows)
        self.assertEqual(len(rows), 14)
        self.assertEqual(
            sum(
                row["data_format"] == "W4A4" and row["method"] == "Ultra-DSP"
                for row in rows
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
