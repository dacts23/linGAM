"""Formula string parser for GAM terms."""

import ast
import re
from typing import Any, Dict, List


def parse_formula(formula: str) -> List[Dict[str, Any]]:
    """Parse a GAM formula string into a list of term configuration dicts.

    Supported syntax::

        s(feature, n_splines=10, lam=1.0)
        te(feature, feature, n_splines=5, lam=0.6)
        te(feature, feature, n_splines=[5, 8])
        f(feature, lam=0.5, coding='one-hot')
        l(feature, lam=0.1)

    Terms are separated by ``+``. Positional arguments are feature indices.
    Keyword arguments are passed through as-is and auto-parsed via
    ``ast.literal_eval``.
    """
    if not formula or not formula.strip():
        raise ValueError("Formula string is empty.")

    stripped = formula.strip()

    chunks = _split_by_plus(stripped)

    result: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        m = re.match(r'(\w+)\s*\(([^)]*)\)', chunk)
        if not m:
            raise ValueError(
                f"Cannot parse term '{chunk}'. Expected e.g. s(0, n_splines=10)."
            )

        func_name = m.group(1).lower()
        if func_name not in ('s', 'te', 'f', 'l'):
            raise ValueError(
                f"Unknown term type '{func_name}'. Expected s, te, f, or l."
            )

        args_str = m.group(2).strip()
        arg_parts = _split_top_level_commas(args_str) if args_str else []

        positional: List[Any] = []
        kwargs: Dict[str, Any] = {}
        for part in arg_parts:
            part = part.strip()
            if '=' in part:
                key, val = part.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    kwargs[key] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    kwargs[key] = val
            else:
                try:
                    positional.append(ast.literal_eval(part))
                except (ValueError, SyntaxError):
                    positional.append(part)

        if func_name in ('s', 'f', 'l'):
            if len(positional) < 1:
                raise ValueError(
                    f"'{func_name}()' requires at least one feature index."
                )
            features = [positional[0]]
        else:
            if len(positional) < 2:
                raise ValueError(
                    f"'te()' requires at least two feature indices."
                )
            features = list(positional)

        result.append({
            'type': func_name,
            'features': features,
            'kwargs': kwargs,
        })

    return result


def _split_by_plus(s: str) -> List[str]:
    """Split on ``+`` that are not inside parentheses."""
    chunks: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '+' and depth == 0:
            chunks.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        chunks.append(''.join(current))
    return chunks


def _split_top_level_commas(s: str) -> List[str]:
    """Split on commas that are not inside brackets ``[ ]``."""
    parts: List[str] = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts
