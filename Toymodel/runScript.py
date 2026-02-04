import main
import numpy as np

activationFunctions = ["tanh"]
neuronsNumber = [70]  # np.arange(30, 90, 20)
epochsNumber = [2000]  # np.arange(1000, 20000, 4000)
examples = [2000]  # np.arange(100, 5100, 500)
seeds = [450]

scenarios = [
    (a, n, e, example, seed)
    for a in activationFunctions
    for n in neuronsNumber
    for e in epochsNumber
    for example in examples
    for seed in seeds
]

for scenario in scenarios:
    script.trainAndEvaluateNetwork(
        scenario[1], scenario[0], scenario[2], scenario[3], scenario[4]
    )
