"""Generic traversal of the AST: a node's children, a pre-order walk, and a bottom-up rewrite."""
from dataclasses import fields, replace
from typing import Callable, Iterator

from parser.grammar import Node


def children(node: Node) -> Iterator[Node]:
    """The direct child nodes of `node`, in field order."""
    for field in fields(node):
        value = getattr(node, field.name)
        for item in value if isinstance(value, tuple) else (value,):
            if isinstance(item, Node):
                yield item


def walk(node: Node) -> Iterator[Node]:
    """Pre-order traversal."""
    yield node
    for child in children(node):
        yield from walk(child)


def map_children(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Rebuild `node` with `fn` applied to each direct child."""
    changes = {}
    for field in fields(node):
        value = getattr(node, field.name)
        if isinstance(value, Node):
            changes[field.name] = fn(value)
        elif isinstance(value, tuple):
            changes[field.name] = tuple(fn(item) if isinstance(item, Node) else item for item in value)
    return replace(node, **changes)


def rewrite(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Bottom-up rewrite: children first, then `fn` on the rebuilt node."""
    return fn(map_children(node, lambda child: rewrite(child, fn)))
