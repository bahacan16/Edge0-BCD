"""The part of shapely this app needs, backed by Clipper2 instead of GEOS.

Real shapely cannot run on iOS: it is a wrapper around GEOS, GEOS is C++, and
a compiled Python extension that was not built into the app will not load. But
the DXF -> PIM script uses a small and specific part of it — twelve calls, of
which exactly two are hard — so the rest is ordinary plane geometry written
here in Python, where it can be read and corrected.

Deliberately not a general shapely. Anything absent is absent because the
script does not call it; adding a stub that returns something plausible would
be worse than an AttributeError naming the gap.
"""

from .geometry import (  # noqa: F401
    GeometryCollection,
    LinearRing,
    MultiPolygon,
    Point,
    Polygon,
)

__version__ = "0.1-edge0"
