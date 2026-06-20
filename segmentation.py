import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries


def choose_channel(image, image_path):
    name = image_path.lower()

    if "olig2" in name:
        return image[:, :, 1], "green"

    if "neun" in name:
        return image[:, :, 2], "blue"

    red_score = image[:, :, 0].max()
    green_score = image[:, :, 1].max()
    blue_score = image[:, :, 2].max()

    scores = [red_score, green_score, blue_score]

    best_channel = scores.index(max(scores))
    if best_channel == 0:
        return image[:, :, 0], "red"
    elif best_channel == 1:
        return image[:, :, 1], "green"
    else:
        return image[:, :, 2], "blue"


def segment_cells(image, image_path):
    gray, used_channel = choose_channel(image, image_path)

    smooth = filters.gaussian(gray, sigma=1)

    if "neun" in image_path.lower():
        threshold_factor = 0.85
        min_size = 45
        min_distance = 7
    elif "olig2" in image_path.lower():
        threshold_factor = 0.95
        min_size = 20
        min_distance = 10
    else:
        threshold_factor = 1.0
        min_size = 30
        min_distance = 8

    threshold = filters.threshold_otsu(smooth) * threshold_factor
    binary = smooth > threshold

    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=30)

    distance = ndi.distance_transform_edt(binary)
    peaks = peak_local_max(distance, min_distance=min_distance, labels=binary)

    markers = np.zeros_like(distance, dtype=int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)

    labels = segmentation.watershed(-distance, markers, mask=binary)

    cells = measure.regionprops(labels)
    count = len(cells)

    return labels, count, used_channel


def make_overlay(image, labels):
    display = image[:, :, :3].astype(float) / 255.0
    boundary = find_boundaries(labels, mode="outer")

    overlay = display.copy()
    overlay[boundary] = [1, 0, 0]

    return display, overlay
