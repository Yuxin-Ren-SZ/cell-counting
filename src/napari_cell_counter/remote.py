"""Remote segmentation via the HuggingFace ``mouseland/cellpose`` Space.

The Space is a Gradio app running Cellpose-SAM on ZeroGPU. We drive it with
``gradio_client``. Verified API (via the Space's ``/config``):

    api_name = "/cellpose_segment"
    inputs   = [file_list, resize, max_iter, flow_threshold, cellprob_threshold]
    outputs  = [outlines_png, flows_png, masks_tif_download, outlines_png_download]

We upload one 2D crop, read back the masks TIF (output index 2), and count.
Remote is **2D only** and exposes only the four numeric params above -- true
3D and the other local eval params are local-only.

The Space resizes masks to the ``resize`` cap, so the returned label array may
be smaller than the crop; we nearest-neighbor upscale it back to the crop's
Y/X shape for pasting into the composite (counts are resolution-independent).

An optional HF token grants more ZeroGPU quota; it is passed through to the
client and never logged or stored in ``run_meta``.
"""
from __future__ import annotations

import glob
import os
import tempfile
import zipfile

import numpy as np

SPACE_ID = "mouseland/cellpose"
API_NAME = "/cellpose_segment"


class RemoteError(RuntimeError):
    """Raised when the remote Space call fails (network, quota, bad output)."""


def _result_path(value) -> str:
    """gradio_client returns file outputs as a local path (str) or a dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("path", "name", "value"):
            if value.get(key):
                return value[key]
    if isinstance(value, (list, tuple)) and value:
        return _result_path(value[0])
    raise RemoteError(f"could not locate masks file in remote output: {value!r}")


def resolve_token(explicit: str | None = None) -> str | None:
    """Find an HF token: explicit arg > cached login > env var.

    Resolution order:
      1. ``explicit`` (what the user typed into the widget), if non-empty;
      2. a cached login from ``huggingface-cli login`` (``huggingface_hub``);
      3. the ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment variables.
    Returns ``None`` if nothing is found (anonymous access).
    """
    if explicit:
        return explicit
    try:
        from huggingface_hub import get_token  # handles cache file + env

        tok = get_token()
        if tok:
            return tok
    except Exception:  # noqa: BLE001 - huggingface_hub optional / old version
        pass
    import os

    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


def _make_client(space: str, token: str | None):
    """Construct a gradio_client Client, tolerant of the token-param rename.

    gradio_client renamed the auth kwarg from ``hf_token`` (older) to ``token``
    (>= ~1.x). We pass the token under whichever name the installed version
    actually accepts.
    """
    import inspect

    from gradio_client import Client

    kwargs = {}
    if token:
        params = inspect.signature(Client.__init__).parameters
        if "hf_token" in params:
            kwargs["hf_token"] = token
        elif "token" in params:
            kwargs["token"] = token
    return Client(space, **kwargs)


def _read_masks(path: str) -> np.ndarray:
    import tifffile

    if path.lower().endswith(".zip"):
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(d)
            tifs = sorted(glob.glob(os.path.join(d, "**", "*.tif"), recursive=True))
            if not tifs:
                raise RemoteError("remote returned a zip with no .tif masks")
            return np.asarray(tifffile.imread(tifs[0]))
    return np.asarray(tifffile.imread(path))


def filter_labels(
    labels: np.ndarray, min_size: int = 0, max_size_fraction: float = 1.0
) -> np.ndarray:
    """Drop labels outside the size range, mirroring what local Cellpose does
    internally (the HF Space returns unfiltered masks, so a spurious oversized
    "blob" or tiny junk can survive without this).

    Removes labels with area ``< min_size`` px or ``> max_size_fraction`` of the
    total image area. Returns a new array with surviving labels' ids preserved.
    """
    if labels.size == 0:
        return labels
    ids, areas = np.unique(labels[labels > 0], return_counts=True)
    if ids.size == 0:
        return labels
    max_area = max_size_fraction * labels.size
    keep = ids[(areas >= max(0, min_size)) & (areas <= max_area)]
    if keep.size == ids.size:
        return labels
    return np.where(np.isin(labels, keep), labels, 0).astype(labels.dtype)


def _resize_labels(labels: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == out_shape:
        return labels
    from skimage.transform import resize

    return resize(
        labels, out_shape, order=0, preserve_range=True, anti_aliasing=False
    ).astype(np.int32)


def segment_remote(
    plane_2d: np.ndarray,
    *,
    resize: float = 1000,
    max_iter: float = 250,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    hf_token: str | None = None,
    out_shape: tuple[int, int] | None = None,
    space: str = SPACE_ID,
) -> tuple[np.ndarray, int, dict]:
    """Segment a single 2D plane on the HF Space. Returns ``(labels, count,
    run_meta)``. Raises :class:`RemoteError` on any failure."""
    if plane_2d.ndim != 2:
        raise RemoteError(
            f"remote backend is 2D only; got a {plane_2d.ndim}D array. "
            "Select a single channel / projection, or use the local backend."
        )
    try:
        from gradio_client import Client, handle_file
    except ImportError as exc:  # pragma: no cover
        raise RemoteError(
            "gradio_client is not installed. `pip install gradio_client` "
            "or install the [remote] extra."
        ) from exc

    # The Space needs integers for these (iteration count / max pixel size);
    # passing floats raises a server-side TypeError.
    max_iter = int(max_iter)
    # "max resize" is a downscale cap -- clamp to the image's own max dimension
    # so a small image is never upscaled (upscaling degrades Cellpose-SAM and
    # can produce spurious merged "blob" masks).
    resize = min(int(resize), max(plane_2d.shape))

    token = resolve_token(hf_token)
    tmpdir = tempfile.mkdtemp(prefix="ccnt_remote_")
    in_path = os.path.join(tmpdir, "crop.tif")
    try:
        import tifffile

        tifffile.imwrite(in_path, plane_2d)

        client = _make_client(space, token)
        try:
            result = client.predict(
                [handle_file(in_path)],
                resize,
                max_iter,
                flow_threshold,
                cellprob_threshold,
                api_name=API_NAME,
            )
        except Exception:
            # Fall back to positional fn_index if the api_name changes.
            result = client.predict(
                [handle_file(in_path)],
                resize,
                max_iter,
                flow_threshold,
                cellprob_threshold,
                fn_index=4,
            )

        if not isinstance(result, (list, tuple)) or len(result) < 3:
            raise RemoteError(f"unexpected remote output shape: {result!r}")
        labels = _read_masks(_result_path(result[2]))
    except RemoteError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any client/network error
        raise RemoteError(f"remote segmentation failed: {exc}") from exc
    finally:
        try:
            os.remove(in_path)
            os.rmdir(tmpdir)
        except OSError:
            pass

    if out_shape is not None:
        labels = _resize_labels(labels, out_shape)
    # The Space returns unfiltered masks; apply the same size limits local
    # Cellpose applies internally so oversized "blob" / tiny junk masks are
    # dropped at the resolution the user actually sees.
    labels = filter_labels(labels, min_size=min_size,
                           max_size_fraction=max_size_fraction)
    count = int(np.unique(labels[labels > 0]).size)

    run_meta = {
        "backend": "remote",
        "space": space,
        "authenticated": bool(token),
        "params": {
            "resize": resize,
            "max_iter": max_iter,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "min_size": min_size,
            "max_size_fraction": max_size_fraction,
        },
    }
    return labels, count, run_meta
