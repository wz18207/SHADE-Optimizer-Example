#include "SHADE.h"

SHADE::SHADE(size_t population_size, size_t dimension,
	const std::vector<double>& min_bound, const std::vector<double>& max_bound,
	size_t max_iter)
	: population_size(population_size), dimension(dimension),
	min_bound(min_bound), max_bound(max_bound), max_iter(max_iter),
	best_individual(dimension) {
	memory_F.resize(5, 0.5);
	memory_CR.resize(5, 0.5);
}

void SHADE::initializePopulation() {
	std::random_device rd;
	std::mt19937 gen(rd());

	for (size_t i = 0; i < population_size; ++i) {
		Individual individual(dimension);
		for (size_t j = 0; j < dimension; ++j) {
			individual.position[j] = generateRandom(min_bound[j], max_bound[j]);
		}
		population.push_back(individual);
	}
}

double SHADE::generateRandom(double min, double max) {
	std::uniform_real_distribution<> dis(min, max);
	return dis(std::mt19937(std::random_device{}()));
}

std::vector<double> SHADE::mutate(const std::vector<double>& target, size_t target_idx) {
	size_t r1, r2, r3;
	do { r1 = generateRandom(0, population_size - 1); } while (r1 == target_idx);
	do { r2 = generateRandom(0, population_size - 1); } while (r2 == target_idx || r2 == r1);
	do { r3 = generateRandom(0, population_size - 1); } while (r3 == target_idx || r3 == r1 || r3 == r2);

	std::vector<double> mutant(dimension);
	for (size_t i = 0; i < dimension; ++i) {
		mutant[i] = population[r1].position[i] + memory_F[0] * (population[r2].position[i] - population[r3].position[i]);
		mutant[i] = std::max(min_bound[i], std::min(max_bound[i], mutant[i]));  // Clamp to bounds
	}
	return mutant;
}

std::vector<double> SHADE::crossover(const std::vector<double>& target, const std::vector<double>& donor, double CR) {
	std::vector<double> offspring(dimension);
	for (size_t i = 0; i < dimension; ++i) {
		offspring[i] = (generateRandom(0, 1) < CR) ? donor[i] : target[i];
	}
	return offspring;
}

void SHADE::optimize(const std::function<double(const std::vector<double>&)>& loss_function) {
	initializePopulation();

	// Calculate fitness for each individual
	for (auto& individual : population) {
		individual.fitness = loss_function(individual.position);
		if (individual.fitness < best_individual.fitness) {
			best_individual = individual;
		}
	}

	// Main loop for optimization
	for (size_t iter = 0; iter < max_iter; ++iter) {
		for (size_t i = 0; i < population_size; ++i) {
			std::vector<double> donor = mutate(population[i].position, i);
			std::vector<double> offspring = crossover(population[i].position, donor, memory_CR[0]);

			double offspring_fitness = loss_function(offspring);
			if (offspring_fitness < population[i].fitness) {
				population[i].position = offspring;
				population[i].fitness = offspring_fitness;

				if (offspring_fitness < best_individual.fitness) {
					best_individual = population[i];
				}
			}
		}
	}
}
