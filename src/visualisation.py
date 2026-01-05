import matplotlib.pyplot as plt

def plot_carbon_impact(carbon_loss, co2_emission):
    labels = ["Carbon Loss", "CO₂ Emission"]
    values = [carbon_loss, co2_emission]

    plt.bar(labels, values, color=["green", "red"])
    plt.title("Carbon Impact of Deforestation")
    plt.ylabel("Tons")
    plt.show()