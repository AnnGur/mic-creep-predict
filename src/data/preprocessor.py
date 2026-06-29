"""MIC value parsing and censoring handling."""

import re
from typing import Optional, Tuple


class MICPreprocessor:
    @staticmethod
    def parse_censored_mic(raw: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse a raw MIC string into (numeric_value, operator).

        Rules (per CLAUDE.md):
          ">8"   -> (16.0, ">")    next doubling dilution
          "<=0.5" -> (0.25, "<=")  half the boundary value
          "0.25"  -> (0.25, None)  plain numeric, no censoring
        """
        if raw is None:
            return None, None
        s = str(raw).strip()
        if not s or s.lower() in ("nan", ""):
            return None, None

        match = re.match(r'^([<>]=?)\s*([0-9]*\.?[0-9]+)$', s)
        if match:
            op, val = match.group(1), float(match.group(2))
            if op in (">", ">="):
                # replace with next doubling dilution
                numeric = val * 2
            else:
                # "<" or "<=" - replace with half the boundary
                numeric = val / 2
            return numeric, op

        # plain numeric
        try:
            return float(s), None
        except ValueError:
            return None, None
