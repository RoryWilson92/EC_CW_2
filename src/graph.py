import matplotlib.pyplot as plt
import pandas as pd

lga = pd.read_csv("../short_lga_out.csv")
ga = pd.read_csv("../short_ga_out.csv")
hc = pd.read_csv("../short_hc_out.csv")

plt.rcParams["font.family"] = "CMU Serif"

plt.figure(figsize=(6, 5), dpi=400)

plt.plot(lga, label="GA linkage learning")
plt.plot(ga, label="Genetic Algorithm")
plt.plot(hc, label="Hill Climber")

plt.xlabel("Generations (evaluations / 1000)")
plt.ylabel("Average Fitness")
plt.title("Average fitness of individuals per generation (Shuffled linkage)")
plt.legend()

plt.tight_layout(pad=0.25)

plt.savefig("../short_graph.png")
