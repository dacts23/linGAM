"""Formula string parser for GAM terms."""

import ast
from typing import Any, Dict, List


def _eval_ast(node: ast.AST) -> Any:
    return node.id if isinstance(node, ast.Name) else ast.literal_eval(node)


def _flatten_plus(node: ast.AST) -> List[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_plus(node.left) + _flatten_plus(node.right)
    return [node]


def parse_formula(formula: str) -> List[Dict[str, Any]]:
    """Parse a GAM formula string into a list of term configuration dicts."""
    if not formula or not formula.strip():
        raise ValueError("Formula string is empty.")
    try:
        tree = ast.parse(formula.strip(), mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse formula '{formula}'.") from exc

    result: List[Dict[str, Any]] = []
    for node in _flatten_plus(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            raise ValueError(f"Cannot parse term in '{formula}'.")
        func_name = node.func.id.lower()
        if func_name not in ("s", "te", "f", "l"):
            raise ValueError(
                f"Unknown term type '{func_name}'. Expected s, te, f, or l."
            )
        positional = [_eval_ast(a) for a in node.args]
        kwargs = {kw.arg: _eval_ast(kw.value) for kw in node.keywords if kw.arg}
        min_feats = 2 if func_name == "te" else 1
        if len(positional) < min_feats:
            raise ValueError(
                f"'{func_name}()' requires at least {min_feats} feature index(es)."
            )
        result.append(
            {
                "type": func_name,
                "features": positional[:1] if min_feats == 1 else list(positional),
                "kwargs": kwargs,
            }
        )
    return result
