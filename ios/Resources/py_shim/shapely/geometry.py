"""Polygons, and the operations on them the script asks for.

Two of these go to Clipper2 through `_edge0geom`: offsetting a boundary
(cutter-radius compensation) and union (which is also how a ring that crosses
itself is repaired). Everything else — area, bounds, centroid, containment,
distance, validity — is plane geometry and lives here.

Rings are held open: the closing point is not stored, and `coords` puts it
back, which is what shapely hands out.
"""

import math

try:
    import _edge0geom
except ImportError:  # pragma: no cover - only on a desktop without the module
    _edge0geom = None


class ShapelyError(Exception):
    pass


def _require_engine():
    if _edge0geom is None:
        raise ShapelyError(
            "_edge0geom is not available; offsetting and union need the "
            "Clipper2 module built into the app"
        )
    return _edge0geom


# ---------------------------------------------------------------- ring helpers


def _clean(points):
    """An open ring of (x, y) floats, with the closing point and any
    consecutive duplicates removed."""
    ring = []
    for point in points:
        x = float(point[0])
        y = float(point[1])
        if ring and abs(ring[-1][0] - x) < 1e-12 and abs(ring[-1][1] - y) < 1e-12:
            continue
        ring.append((x, y))
    while len(ring) > 1 and abs(ring[0][0] - ring[-1][0]) < 1e-12 \
            and abs(ring[0][1] - ring[-1][1]) < 1e-12:
        ring.pop()
    return ring


def _signed_area(ring):
    total = 0.0
    count = len(ring)
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _ring_bounds(ring):
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def _ring_centroid(ring):
    """Area-weighted centroid and signed area. Falls back to the vertex mean
    for a degenerate ring, which is what a zero-area sliver leaves."""
    count = len(ring)
    cx = cy = twice = 0.0
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        cross = x1 * y2 - x2 * y1
        twice += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area = twice / 2.0
    if abs(area) < 1e-12:
        return (sum(p[0] for p in ring) / count, sum(p[1] for p in ring) / count, 0.0)
    return (cx / (3.0 * twice), cy / (3.0 * twice), area)


def _oriented(ring, counter_clockwise):
    if (_signed_area(ring) >= 0.0) == counter_clockwise:
        return ring
    return ring[::-1]


def _point_in_ring(x, y, ring, boundary=True):
    """Crossing number. `boundary` is the answer for a point lying exactly on
    an edge, which is a separate question from being inside: shapely's
    `contains` excludes the boundary, while deciding which outer ring owns a
    hole wants a touching point to count."""
    inside = False
    count = len(ring)
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return boundary
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing > x:
                inside = not inside
    return inside


def _point_on_segment(px, py, x1, y1, x2, y2, tolerance=1e-9):
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) > tolerance * max(1.0, abs(x2 - x1) + abs(y2 - y1)):
        return False
    return (min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance
            and min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance)


def _segments_cross(a, b, c, d):
    """Proper or improper intersection of segments ab and cd."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = side(c, d, a), side(c, d, b)
    d3, d4 = side(a, b, c), side(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    for p, q, r in ((c, d, a), (c, d, b), (a, b, c), (a, b, d)):
        if _point_on_segment(r[0], r[1], p[0], p[1], q[0], q[1]):
            return True
    return False


def _segment_distance(a, b, c, d):
    if _segments_cross(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def _point_segment_distance(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _ring_self_intersects(ring):
    """Whether any two non-adjacent edges of the ring meet.

    Quadratic, with a bounding-box rejection that removes almost every pair on
    a real outline. A CAD part is tens to hundreds of points, so this is a
    millisecond; it is not meant for a ring of ten thousand.
    """
    count = len(ring)
    if count < 4:
        return False
    edges = [(ring[i], ring[(i + 1) % count]) for i in range(count)]
    boxes = [
        (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        for a, b in edges
    ]
    for i in range(count):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, count):
            # Adjacent edges share a vertex by construction, and the first and
            # last edge close the ring.
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            if _segments_cross(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return True
    return False


# ------------------------------------------------------------------ geometries


class Point:
    geom_type = "Point"

    __slots__ = ("x", "y")

    def __init__(self, x, y=None):
        if y is None:
            x, y = x[0], x[1]
        self.x = float(x)
        self.y = float(y)

    @property
    def coords(self):
        return [(self.x, self.y)]

    @property
    def bounds(self):
        return (self.x, self.y, self.x, self.y)

    @property
    def area(self):
        return 0.0

    @property
    def is_empty(self):
        return False

    def distance(self, other):
        if isinstance(other, Point):
            return math.hypot(self.x - other.x, self.y - other.y)
        return other.distance(self)

    def __repr__(self):
        return "POINT (%g %g)" % (self.x, self.y)


class LinearRing:
    geom_type = "LinearRing"

    __slots__ = ("_ring",)

    def __init__(self, ring):
        self._ring = _clean(ring)

    @property
    def coords(self):
        """Closed, the way shapely hands it out: the first point repeats."""
        if not self._ring:
            return []
        return list(self._ring) + [self._ring[0]]

    @property
    def bounds(self):
        return _ring_bounds(self._ring) if self._ring else ()

    def __len__(self):
        return len(self.coords)

    def __iter__(self):
        return iter(self.coords)


class Polygon:
    geom_type = "Polygon"

    __slots__ = ("_shell", "_holes")

    def __init__(self, shell=None, holes=None):
        if shell is None:
            self._shell = []
        elif isinstance(shell, Polygon):
            self._shell = list(shell._shell)
            holes = holes if holes is not None else [list(h) for h in shell._holes]
        elif isinstance(shell, LinearRing):
            self._shell = _clean(shell.coords)
        else:
            self._shell = _clean(shell)
        self._holes = [_clean(h.coords if isinstance(h, LinearRing) else h)
                       for h in (holes or [])]
        self._holes = [h for h in self._holes if len(h) >= 3]

    # -- description

    @property
    def is_empty(self):
        return len(self._shell) < 3

    @property
    def exterior(self):
        return LinearRing(self._shell)

    @property
    def interiors(self):
        return [LinearRing(h) for h in self._holes]

    @property
    def area(self):
        if self.is_empty:
            return 0.0
        return abs(_signed_area(self._shell)) - sum(abs(_signed_area(h)) for h in self._holes)

    @property
    def bounds(self):
        if self.is_empty:
            return ()
        return _ring_bounds(self._shell)

    @property
    def centroid(self):
        if self.is_empty:
            return Point(0.0, 0.0)
        cx, cy, area = _ring_centroid(self._shell)
        if not self._holes or area == 0.0:
            return Point(cx, cy)
        # Holes pull the centroid: subtract each one's moment.
        total = abs(area)
        mx = cx * total
        my = cy * total
        for hole in self._holes:
            hx, hy, harea = _ring_centroid(hole)
            total -= abs(harea)
            mx -= hx * abs(harea)
            my -= hy * abs(harea)
        if abs(total) < 1e-12:
            return Point(cx, cy)
        return Point(mx / total, my / total)

    @property
    def is_valid(self):
        if self.is_empty:
            return False
        if _ring_self_intersects(self._shell):
            return False
        return not any(_ring_self_intersects(h) for h in self._holes)

    def representative_point(self):
        """A point guaranteed to be inside, which the centroid of a crescent
        is not. Scans one horizontal line and takes the middle of its widest
        interior span."""
        if self.is_empty:
            return Point(0.0, 0.0)
        centroid = self.centroid
        if self.contains(centroid):
            return centroid
        minx, miny, maxx, maxy = self.bounds
        y = (miny + maxy) / 2.0
        crossings = []
        for ring in [self._shell] + self._holes:
            count = len(ring)
            for i in range(count):
                x1, y1 = ring[i]
                x2, y2 = ring[(i + 1) % count]
                if (y1 > y) != (y2 > y):
                    crossings.append(x1 + (y - y1) / (y2 - y1) * (x2 - x1))
        crossings.sort()
        best = None
        for i in range(0, len(crossings) - 1, 2):
            width = crossings[i + 1] - crossings[i]
            if best is None or width > best[0]:
                best = (width, (crossings[i] + crossings[i + 1]) / 2.0)
        if best is None:
            return centroid
        return Point(best[1], y)

    # -- relations

    def contains(self, other):
        if self.is_empty:
            return False
        if isinstance(other, Point):
            # Strictly inside, as shapely means it: a point on the edge is on
            # the boundary, not in the interior. `representative_point` leans
            # on this — the centroid of an L sits exactly on its inner edge,
            # and taking it would hand back a point that is not in the part.
            if not _point_in_ring(other.x, other.y, self._shell, boundary=False):
                return False
            return not any(_point_in_ring(other.x, other.y, h) for h in self._holes)
        if isinstance(other, Polygon):
            if other.is_empty:
                return False
            if not all(self.contains(Point(p)) for p in other._shell):
                return False
            return not _rings_cross(self._shell, other._shell)
        raise TypeError("contains() takes a Point or a Polygon")

    def distance(self, other):
        if isinstance(other, Point):
            if self.contains(other):
                return 0.0
            return min(
                _point_segment_distance(
                    (other.x, other.y), ring[i], ring[(i + 1) % len(ring)])
                for ring in [self._shell] + self._holes
                for i in range(len(ring))
            )
        if isinstance(other, MultiPolygon):
            return min((self.distance(g) for g in other.geoms), default=float("inf"))
        if not isinstance(other, Polygon):
            raise TypeError("distance() takes a Point, Polygon or MultiPolygon")
        if self.is_empty or other.is_empty:
            return float("inf")
        # Overlapping is distance zero, and containment counts as overlapping.
        if _rings_cross(self._shell, other._shell):
            return 0.0
        if _point_in_ring(other._shell[0][0], other._shell[0][1], self._shell) \
                or _point_in_ring(self._shell[0][0], self._shell[0][1], other._shell):
            return 0.0
        return _rings_distance(self._shell, other._shell)

    # -- operations

    def buffer(self, distance, resolution=16, quad_segs=None, cap_style=1,
               join_style=1, mitre_limit=5.0, single_sided=False):
        """Offset the boundary outwards (positive) or inwards (negative).

        `buffer(0)` is shapely's repair idiom rather than an offset, and is
        answered with a self-union — which is what resolves a ring that
        crosses itself.
        """
        engine = _require_engine()
        rings = self._clipper_rings()
        if not rings:
            return Polygon()
        if distance == 0:
            return _from_paths(engine.union(rings, None, "nonzero"))

        segments = quad_segs if quad_segs is not None else resolution
        joins = {1: "round", 2: "miter", 3: "bevel"}
        # Clipper measures arc quality as the largest allowed deviation from
        # the true arc; shapely counts segments per quarter circle. For a
        # radius of |distance| the two are the same statement.
        tolerance = 0.0
        if segments and segments > 0:
            tolerance = abs(distance) * (1.0 - math.cos(math.pi / (4.0 * segments)))
        return _from_paths(
            engine.offset(
                rings, float(distance), join=joins.get(join_style, "round"),
                miter_limit=float(mitre_limit), arc_tolerance=tolerance))

    def union(self, other):
        engine = _require_engine()
        return _from_paths(
            engine.union(self._clipper_rings(), _clipper_rings_of(other), "nonzero"))

    def _clipper_rings(self):
        """Shell counter-clockwise, holes clockwise: Clipper reads a ring's
        orientation to decide which way an offset goes."""
        if self.is_empty:
            return []
        return [_oriented(self._shell, True)] + [_oriented(h, False) for h in self._holes]

    def __repr__(self):
        return "POLYGON (%d points, area %g)" % (len(self._shell), self.area)


class MultiPolygon:
    geom_type = "MultiPolygon"

    __slots__ = ("_geoms",)

    def __init__(self, polygons=None):
        self._geoms = [p for p in (polygons or []) if not p.is_empty]

    @property
    def geoms(self):
        return list(self._geoms)

    def __len__(self):
        return len(self._geoms)

    def __iter__(self):
        return iter(self._geoms)

    @property
    def is_empty(self):
        return not self._geoms

    @property
    def area(self):
        return sum(g.area for g in self._geoms)

    @property
    def bounds(self):
        if not self._geoms:
            return ()
        boxes = [g.bounds for g in self._geoms]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    @property
    def centroid(self):
        total = self.area
        if total == 0.0:
            return Point(0.0, 0.0)
        x = sum(g.centroid.x * g.area for g in self._geoms) / total
        y = sum(g.centroid.y * g.area for g in self._geoms) / total
        return Point(x, y)

    @property
    def is_valid(self):
        return all(g.is_valid for g in self._geoms)

    def representative_point(self):
        largest = max(self._geoms, key=lambda g: g.area, default=None)
        return largest.representative_point() if largest else Point(0.0, 0.0)

    def contains(self, other):
        return any(g.contains(other) for g in self._geoms)

    def distance(self, other):
        return min((g.distance(other) for g in self._geoms), default=float("inf"))

    def buffer(self, distance, **keywords):
        engine = _require_engine()
        rings = _clipper_rings_of(self)
        if not rings:
            return Polygon()
        if distance == 0:
            return _from_paths(engine.union(rings, None, "nonzero"))
        segments = keywords.get("quad_segs") or keywords.get("resolution", 16)
        joins = {1: "round", 2: "miter", 3: "bevel"}
        tolerance = abs(distance) * (1.0 - math.cos(math.pi / (4.0 * segments))) \
            if segments else 0.0
        return _from_paths(
            engine.offset(
                rings, float(distance),
                join=joins.get(keywords.get("join_style", 1), "round"),
                miter_limit=float(keywords.get("mitre_limit", 5.0)),
                arc_tolerance=tolerance))

    def union(self, other):
        engine = _require_engine()
        return _from_paths(
            engine.union(_clipper_rings_of(self), _clipper_rings_of(other), "nonzero"))

    def __repr__(self):
        return "MULTIPOLYGON (%d parts, area %g)" % (len(self._geoms), self.area)


class GeometryCollection:
    """Present so `geom_type` comparisons have something to fail against
    rather than an AttributeError. Nothing here produces one."""

    geom_type = "GeometryCollection"

    def __init__(self, geoms=None):
        self.geoms = list(geoms or [])

    @property
    def is_empty(self):
        return not self.geoms


# -------------------------------------------------------------- shared helpers


def _rings_cross(a, b):
    for i in range(len(a)):
        p1, p2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            q1, q2 = b[j], b[(j + 1) % len(b)]
            if _segments_cross(p1, p2, q1, q2):
                return True
    return False


def _rings_distance(a, b):
    best = float("inf")
    for i in range(len(a)):
        p1, p2 = a[i], a[(i + 1) % len(a)]
        for j in range(len(b)):
            q1, q2 = b[j], b[(j + 1) % len(b)]
            gap = _segment_distance(p1, p2, q1, q2)
            if gap < best:
                best = gap
                if best == 0.0:
                    return 0.0
    return best


def _clipper_rings_of(geometry):
    if geometry is None:
        return []
    if isinstance(geometry, Polygon):
        return geometry._clipper_rings()
    if isinstance(geometry, MultiPolygon):
        rings = []
        for part in geometry.geoms:
            rings.extend(part._clipper_rings())
        return rings
    raise TypeError("expected a Polygon or MultiPolygon")


def _from_paths(paths):
    """Clipper returns a flat list of rings. Positive area is an outer
    boundary, negative is a hole, and a hole belongs to the smallest outer
    ring that contains it."""
    outers = []
    holes = []
    for path in paths:
        ring = _clean(path)
        if len(ring) < 3:
            continue
        (outers if _signed_area(ring) > 0 else holes).append(ring)

    if not outers:
        return Polygon()

    assigned = [[] for _ in outers]
    order = sorted(range(len(outers)), key=lambda i: abs(_signed_area(outers[i])))
    for hole in holes:
        x, y = hole[0]
        for index in order:  # smallest first, so the innermost container wins
            if _point_in_ring(x, y, outers[index]):
                assigned[index].append(hole)
                break

    polygons = [Polygon(outers[i], assigned[i]) for i in range(len(outers))]
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)
