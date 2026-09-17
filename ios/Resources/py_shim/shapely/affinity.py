"""Affine transforms, the three the script uses.

Real shapely dispatches these through GEOS; there is nothing to dispatch. A
translate is an addition and a rotate is two multiplications, so these are
written out rather than routed anywhere.
"""

import math

from .geometry import MultiPolygon, Point, Polygon


def _resolve_origin(geometry, origin):
    if origin == "center":
        minx, miny, maxx, maxy = geometry.bounds
        return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    if origin == "centroid":
        centre = geometry.centroid
        return (centre.x, centre.y)
    if isinstance(origin, Point):
        return (origin.x, origin.y)
    return (float(origin[0]), float(origin[1]))


def _map(geometry, function):
    if isinstance(geometry, Point):
        return Point(*function(geometry.x, geometry.y))
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([_map(part, function) for part in geometry.geoms])
    if isinstance(geometry, Polygon):
        if geometry.is_empty:
            return Polygon()
        shell = [function(x, y) for x, y in geometry.exterior.coords[:-1]]
        holes = [[function(x, y) for x, y in ring.coords[:-1]]
                 for ring in geometry.interiors]
        return Polygon(shell, holes)
    raise TypeError("expected a Point, Polygon or MultiPolygon")


def translate(geometry, xoff=0.0, yoff=0.0, zoff=0.0):
    return _map(geometry, lambda x, y: (x + xoff, y + yoff))


def rotate(geometry, angle, origin="center", use_radians=False):
    radians = angle if use_radians else math.radians(angle)
    cos = math.cos(radians)
    sin = math.sin(radians)
    ox, oy = _resolve_origin(geometry, origin)

    def turn(x, y):
        dx, dy = x - ox, y - oy
        return (ox + dx * cos - dy * sin, oy + dx * sin + dy * cos)

    return _map(geometry, turn)


def scale(geometry, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
    ox, oy = _resolve_origin(geometry, origin)
    # A negative factor mirrors, and mirroring reverses a ring's orientation.
    # Nothing here depends on that: orientation is normalised again on the way
    # into Clipper.
    return _map(geometry, lambda x, y: (ox + (x - ox) * xfact, oy + (y - oy) * yfact))
