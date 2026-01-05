def calculate_carbon_emission(area_km2, carbon_density=200):
    """
    Calculate carbon loss and CO2 emissions due to deforestation.

    Parameters:
    area_km2 (float): Deforested area in square kilometers
    carbon_density (float): Average carbon density (tons per hectare)

    Returns:
    carbon_loss (float): Carbon loss in tons
    co2_emission (float): CO2 emission in tons
    """

    # Convert area from km² to hectares
    area_hectares = area_km2 * 100

    # Calculate carbon loss
    carbon_loss = area_hectares * carbon_density

    # Convert carbon loss to CO2 emission
    co2_emission = carbon_loss * 3.67

    return carbon_loss, co2_emission


# Optional test (can be removed later)
if __name__ == "__main__":
    area = 2.0  # example deforested area in km²
    carbon, co2 = calculate_carbon_emission(area)

    print("Deforested Area (km²):", area)
    print("Carbon Loss (tons):", carbon)
    print("CO2 Emission (tons):", co2)