import numpy as np

def detect_deforestation(mask_before, mask_after):
    """
    Detect deforested pixels between two masks.

    mask_before: binary mask (earlier year)
    mask_after: binary mask (later year)

    Returns:
    deforestation_mask, deforested_pixel_count
    """
    # Forest → Non-forest
    deforestation_mask = (mask_before == 1) & (mask_after == 0)

    deforested_pixels = np.sum(deforestation_mask)

    return deforestation_mask.astype(int), deforested_pixels
def calculate_deforested_area(deforested_pixels, pixel_area_m2=900):
    """
    Convert pixel count to area.

    Returns:
    area_km2
    """
    area_m2 = deforested_pixels * pixel_area_m2
    area_km2 = area_m2 / 1_000_000
    return area_km2