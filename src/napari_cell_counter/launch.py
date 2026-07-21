"""Launch napari with the Cell Counter widget already docked.

Exposed as the ``napari-cell-counter`` console script (see pyproject) and as
``python -m napari_cell_counter``. The double-clickable launchers call this.
"""
from __future__ import annotations


def main() -> None:
    import napari

    from ._widget import make_cell_counter_widget

    viewer = napari.Viewer()
    widget = make_cell_counter_widget(viewer)
    viewer.window.add_dock_widget(widget, name="Cell Counter", area="right")
    napari.run()


if __name__ == "__main__":
    main()
