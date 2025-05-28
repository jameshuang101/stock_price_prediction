import numpy as np
import pygad
import pygad.nn
import pygad.gann
from stock_prediction.components.dataset import StockDataset
from sklearn.model_selection import train_test_split


def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs
    predictions = pygad.nn.predict(
        last_layer=GANN_instance.population_networks[sol_idx], data_inputs=data_inputs
    )
    correct_predictions = np.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions / data_outputs.size) * 100

    return solution_fitness


def callback_generation(ga_instance):
    global GANN_instance
    population_matrices = pygad.gann.population_as_matrices(
        population_networks=GANN_instance.population_networks,
        population_vectors=ga_instance.population,
    )
    GANN_instance.update_population_trained_weights(
        population_trained_weights=population_matrices
    )
    print(
        "Generation = {generation}".format(generation=ga_instance.generations_completed)
    )
    print("Accuracy   = {fitness}".format(fitness=ga_instance.best_solution()[1]))


trainset = StockDataset(
    stock="AAPL", start_date="2000-01-01", end_date="2025-01-01", inc_macro=False
)
X = trainset.X
y = trainset.y[:, 0]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

GANN_instance = pygad.gann.GANN(
    num_solutions=5,
    num_neurons_input=2,
    num_neurons_hidden_layers=[2],
    num_neurons_output=2,
    hidden_activations=["relu"],
    output_activation="softmax",
)

population_vectors = pygad.gann.population_as_vectors(
    population_networks=GANN_instance.population_networks
)

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=3,
    initial_population=population_vectors.copy(),
    fitness_func=fitness_func,
    mutation_percent_genes=5,
    callback_generation=callback_generation,
)

ga_instance.run()
ga_instance.plot_result()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution)
print(solution_fitness)
print(solution_idx)
