"""
Editable diagram module for Flowchart to Editable Diagram.

Converts detection pipeline output into a NetworkX DiGraph and renders it
via Graphviz DOT format, supporting interactive node/edge editing and exports.

Layout strategy
───────────────
When nodes have pixel coordinates from the pipeline (they always do after
detection), we pin them at those positions using Graphviz's `neato -n`
mode.  This preserves the original spatial layout of the diagram — nodes
appear where they were in the source image, not wherever Graphviz's
auto-layout engine decides to put them.

When a node has no position (e.g. manually added via the editor), it is
placed at [0, 0] and Graphviz will position it automatically relative to
the rest.

Coordinate conversion
─────────────────────
Graphviz uses inches with 72 DPI as its internal unit.  Image pixels are
divided by 72 to get inches.  The Y axis is also flipped (image: top=0,
grows down; Graphviz: bottom=0, grows up) so we subtract from the image
height to invert Y.
"""

from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
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
        # plain rectangles
        "box": "box",
        "rectangle": "box",
        "rect": "box",
        "square": "box",
        # rounded rectangle — most common shape from the detectors; render as
        # a box with the Graphviz "rounded" style (set via node attrs below)
        "rounded_rectangle": "box",
        "rounded": "box",
        # circles / ovals
        "circle": "ellipse",
        "oval": "ellipse",
        "ellipse": "ellipse",
        "stadium": "ellipse",
        "pill": "ellipse",
        # decision diamond
        "diamond": "diamond",
        "rhombus": "diamond",
        "decision": "diamond",
        # data / I-O parallelogram
        "parallelogram": "parallelogram",
        "data": "parallelogram",
        "input": "parallelogram",
        "output": "parallelogram",
        # storage cylinder
        "cylinder": "cylinder",
        "database": "cylinder",
        # hexagon
        "hexagon": "hexagon",
        # generic polygon — render as a plain box; avoids Graphviz "polygon"
        # which requires extra side-count attributes
        "polygon": "box",
    }
    return mapping.get(t, "box")


def _is_rounded(node_type: str) -> bool:
    """True when the shape should use Graphviz's rounded corner style."""
    t = (node_type or "").strip().lower()
    return t in ("rounded_rectangle", "rounded")


def _image_bounds(g: nx.DiGraph) -> Tuple[int, int]:
    """Return (max_x, max_y) across all node bboxes/centers in the graph.
    Used to flip the Y axis when converting pixel → Graphviz coordinates."""
    max_x, max_y = 0, 0
    for _, attrs in g.nodes(data=True):
        bbox = attrs.get("bbox")
        if bbox and len(bbox) == 4:
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
        center = attrs.get("center")
        if center and len(center) == 2:
            max_x = max(max_x, center[0])
            max_y = max(max_y, center[1])
    # Fall back to a sensible default if nothing was found
    return max(max_x, 800), max(max_y, 600)


def _px_to_gv(cx: float, cy: float, img_height: float, scale: float = 72.0) -> str:
    """Convert pixel center (cx, cy) to a Graphviz pos string.

    Graphviz Y=0 is at the bottom; image Y=0 is at the top.
    Dividing by `scale` (72 ≈ default DPI) converts pixels to inches.
    The '!' suffix tells neato to treat the position as fixed (pinned).
    """
    gv_x = cx / scale
    gv_y = (img_height - cy) / scale
    return f"{gv_x:.4f},{gv_y:.4f}!"


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
    Render a NetworkX DiGraph to Graphviz DOT format preserving the spatial
    layout from the original detected pixel positions.

    Uses the `neato` engine with `pos` attributes pinned from each node's
    detected center coordinate, so the diagram mirrors the source image's
    layout instead of being re-arranged by Graphviz's auto-layout.
    """
    _, img_h = _image_bounds(g)

    # neato with overlap=false: respects fixed positions but nudges nodes
    # apart when they overlap, keeping the layout clean.
    dot = graphviz.Digraph(
        name=graph_name,
        comment="Flowchart to Editable Diagram",
        engine="neato",           # position-aware engine
    )
    dot.attr(
        overlap="false",          # separate overlapping nodes while keeping structure
        splines="true",           # smooth curved edges that follow the real paths
        sep="+8",                 # minimum separation between nodes (points)
        bgcolor="transparent",
        outputorder="edgesfirst", # draw edges behind nodes
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

        # Show "ID: label" so nodes are identifiable in the editor
        display_label = f"{node_id}: {label}" if label and label != node_id else str(node_id)

        node_attrs: Dict[str, str] = {
            "label": display_label,
            "shape": gv_shape,
        }

        # Apply rounded corners for rounded_rectangle shapes
        if _is_rounded(raw_type):
            node_attrs["style"] = "filled,rounded"

        # Pin position from detected pixel coordinates.
        # Prefer bbox center for accuracy; fall back to stored center; default 0,0.
        bbox = attrs.get("bbox")
        center = attrs.get("center")
        if bbox and len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
        elif center and len(center) == 2:
            cx, cy = float(center[0]), float(center[1])
        else:
            cx, cy = 0.0, 0.0

        node_attrs["pos"] = _px_to_gv(cx, cy, img_h)

        # Scale node width/height from bbox pixel dimensions.
        # Graphviz uses inches; 1 px ≈ 1/72 inch at default DPI.
        if bbox and len(bbox) == 4:
            w_in = max(0.5, (bbox[2] - bbox[0]) / 72.0)
            h_in = max(0.3, (bbox[3] - bbox[1]) / 72.0)
            node_attrs["width"]      = f"{w_in:.3f}"
            node_attrs["height"]     = f"{h_in:.3f}"
            node_attrs["fixedsize"]  = "false"   # allow label to expand the box

        dot.node(str(node_id), **node_attrs)

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
