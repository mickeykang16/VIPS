"""Adapter that exposes V2XRealMapWrapper with the nuPlan AbstractMap interface
points used by IDM (`get_starting_segment` and friends).

V2XRealMapWrapper has:
  is_in_layer(x: float, y: float, layer)  # x/y separately, only handles ROADBLOCK
  get_all_map_objects(*args, **kwargs)    # returns roadblocks+lanes regardless of layer
  get_proximal_map_objects(point, radius, layers)  # nuPlan-style, lanes etc available

nuPlan's IDM expects:
  is_in_layer(point, layer)               # point has .x/.y attributes
  get_all_map_objects(point, layer)       # returns lane-like objects with .baseline_path

This adapter forwards everything to the wrapped map but rewires those two methods
so IDM can run on V2X-Real lane geometry without modifying V2XRealMapWrapper.
"""
from typing import Any, List

from nuplan.common.maps.abstract_map import SemanticMapLayer


class V2XRealIDMMapAdapter:
    """Wraps V2XRealMapWrapper for nuPlan-IDM consumers.

    Only implements the subset of AbstractMap that IDM actually calls. Delegates
    everything else to the underlying wrapper via __getattr__.
    """

    def __init__(self, v2x_map):
        self._inner = v2x_map

    # Delegate all unknown attribute access to the inner V2XRealMapWrapper
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def map_name(self) -> str:
        return self._inner.map_name

    def is_in_layer(self, point: Any, layer: SemanticMapLayer) -> bool:
        """nuPlan-style: point with .x/.y attrs.

        For LANE: check membership in any lane polygon.
        For INTERSECTION: check membership in any junction polygon.
        Other layers: defer to the inner wrapper (which only handles ROADBLOCK).
        """
        from shapely.geometry import Point as ShapelyPoint

        x, y = float(point.x), float(point.y)
        sp = ShapelyPoint(x, y)

        if layer == SemanticMapLayer.LANE:
            for lane in self._inner._get_lanes():
                if lane.polygon.contains(sp):
                    return True
            return False
        if layer == SemanticMapLayer.INTERSECTION:
            for junction in self._inner._get_junctions():
                if junction.contains(sp):
                    return True
            return False
        # Fallback (e.g. ROADBLOCK) — V2XRealMapWrapper takes (x, y, layer)
        try:
            return self._inner.is_in_layer(x, y, layer)
        except TypeError:
            return False

    def get_all_map_objects(self, point: Any, layer: SemanticMapLayer) -> List[Any]:
        """nuPlan-style: return lane-like objects that contain `point`."""
        from shapely.geometry import Point as ShapelyPoint

        sp = ShapelyPoint(float(point.x), float(point.y))

        if layer == SemanticMapLayer.LANE:
            return [lane for lane in self._inner._get_lanes() if lane.polygon.contains(sp)]
        if layer == SemanticMapLayer.LANE_CONNECTOR:
            connectors = self._inner._get_lane_connectors()
            # Lane connectors may not have a polygon; if so, treat them as "anywhere"
            out = []
            for c in connectors:
                poly = getattr(c, "polygon", None)
                if poly is None or poly.contains(sp):
                    out.append(c)
            return out
        return []
