"""Bilingual document builders — the assembly manual, the sourcing guide and
the extrusion tech drawing.

Standard-library only (no build123d): the manual and the guide consume the
SVG renders the assembly pipeline already produced; the drawing consumes
the model's own constants (lengths, hole specs, profile vertices).
``build_manual``, ``build_sourcing_guide`` and ``build_extrusion_drawing``
are the entry points; ``assets`` / ``common`` / ``paginate`` / ``pdf`` are
their support modules.
"""


class BuildError(Exception):
    """A user-facing build failure: ``main()`` prints the message on its own,
    without a Python traceback."""
