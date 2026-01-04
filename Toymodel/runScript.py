import script
import numpy as np

activationFunctions = ("tanh", "sigmoid")
neuronsNumber = [70]  # np.arange(30, 70, 20)
epochsNumber = [50000]  # np.arange(1000, 20000, 4000)

# scenarios = [
#    (a, n, e) for a in activationFunctions for n in neuronsNumber for e in epochsNumber
# ]

scenarios = [("sigmoid", n, e) for n in neuronsNumber for e in epochsNumber]

for scenario in scenarios:
    script.trainAndEvaluateNetwork(scenario[1], scenario[0], scenario[2])
