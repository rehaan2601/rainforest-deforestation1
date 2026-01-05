import numpy as np
from change_detection import detect_deforestation, calculate_deforested_area
from carbon_calculation import calculate_carbon_emission

# Dummy masks for testing
mask_before = np.random.randint(0, 2, (224, 224))
mask_after = np.random.randint(0, 2, (224, 224))

# Change detection
def_mask, def_pixels = detect_deforestation(mask_before, mask_after)

# Area calculation
area_km2 = calculate_deforested_area(def_pixels)

# Carbon impact
carbon_loss, co2_emission = calculate_carbon_emission(area_km2)

print("Deforested Pixels:", def_pixels)
print("Deforested Area (km²):", area_km2)
print("Carbon Loss (tons):", carbon_loss)
print("CO₂ Emission (tons):", co2_emission)