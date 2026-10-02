"""
Editable diagram module for VisionFlow AI.

Converts detection pipeline output into a NetworkX DiGraph and renders it
via Graphviz DOT format, supporting interactive node/edge editing and exports.
"""

from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Optional
import networkx as nx
import graphviz


def map_type_to_graphviz_shape(node_type: str) -> str:
    """
    Map a node type/shape string to a valid Graphviz shape attribute.
    Defaults to 'box' for unknown/unrecognized shapes.
    """
    if not node_type or not isinstance(node_type, str):
        return "box"

    t = node_type.strip().lower()
    mapping = {
        "box": "box",
        "rectangle": "box",
        "rect": "box",
        "square": "box",
        "circle": "ellipse",
        "oval": "ellipse",
        "ellipse": "ellipse",
        "stadium": "ellipse",
        "pill": "ellipse",
        "diamond": "diamond",
        "rhombus": "diamond",
        "decision": "diamond",
        "parallelogram": "parallelogram",
        "data": "parallelogram",
        "input": "parallelogram",
        "output": "parallelogram",
        "cylinder": "cylinder",
        "database": "cylinder",
        "hexagon": "hexagon",
    }
    return mapping.get(t, "box")


def build_digraph_from_pipeline(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from pipeline nodes and edges lists.
    Preserves label, type (or shape), bbox, confidence, and other attributes.
    """
    g = nx.DiGraph()
    for n in nodes:
        nid = str(n.get("id"))
        if not nid:
            continue
        label = n.get("label", nid)
        node_type = n.get("type") or n.get("shape", "box")
        bbox = n.get("bbox", [0, 0, 0, 0])
        confidence = n.get("confidence", 1.0)

        attrs = dict(n)
        attrs.update({
            "label": label,
            "type": node_type,
            "shape": node_type,
            "bbox": bbox,
            "confidence": confidence,
        })
        g.add_node(nid, **attrs)

    for e in edges:
        src = str(e.get("source"))
        tgt = str(e.get("target"))
        if src in g and tgt in g:
            attrs = dict(e)
            attrs.setdefault("confidence", 1.0)
            g.add_edge(src, tgt, **attrs)

    return g


def render_digraph_to_graphviz(
    g: nx.DiGraph,
    graph_name: str = "WorkflowDiagram",
) -> graphviz.Digraph:
    """
    Render a NetworkX DiGraph to Graphviz DOT format and return a graphviz.Digraph object.
    """
    dot = graphviz.Digraph(name=graph_name, comment="VisionFlow Editable Diagram")
    dot.attr(
        rankdir="TB",
        splines="ortho",
        nodesep="0.5",
        ranksep="0.6",
        bgcolor="transparent",
    )
    dot.attr(
        "node",
        fontname="Helvetica",
        fontsize="11",
        style="filled",
        fillcolor="#1e293b",
        fontcolor="#f8fafc",
        color="#3b82f6",
        penwidth="1.5",
    )
    dot.attr(
        "edge",
        fontname="Helvetica",
        fontsize="9",
        color="#94a3b8",
        fontcolor="#cbd5e1",
        penwidth="1.2",
        arrowsize="0.8",
    )

    for node_id, attrs in g.nodes(data=True):
        label = str(attrs.get("label", node_id))
        raw_type = str(attrs.get("type") or attrs.get("shape", "box"))
        gv_shape = map_type_to_graphviz_shape(raw_type)

        display_label = f"{node_id}: {label}" if label and label != node_id else str(node_id)

        dot.node(
            str(node_id),
            label=display_label,
            shape=gv_shape,
        )

    for src, tgt, attrs in g.edges(data=True):
        edge_label = ""
        if "label" in attrs and attrs["label"]:
            edge_label = str(attrs["label"])
        dot.edge(str(src), str(tgt), label=edge_label)

    return dot


def export_graphviz_to_bytes(dot: graphviz.Digraph, fmt: str = "png") -> bytes:
    """
    Use Graphviz .render() to generate binary bytes (PNG or PDF) from the Digraph object.
    """
    fmt = fmt.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}") as tmp:
        tmp_base = tmp.name

    rendered_file = dot.render(filename=tmp_base, format=fmt, cleanup=True)
    try:
        with open(rendered_file, "rb") as f:
            data = f.read()
    finally:
        if os.path.exists(rendered_file):
            try:
                os.remove(rendered_file)
            except OSError:
                pass
        if os.path.exists(tmp_base):
            try:
                os.remove(tmp_base)
            except OSError:
                pass

    return data
