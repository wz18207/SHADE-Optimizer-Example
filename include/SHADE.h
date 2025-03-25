#pragma once
#ifndef SHADE_H
#define SHADE_H

#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <functional>

// Define an individual in the population
struct Individual {
	std::vector<double> position;
	double fitness;

	Individual(size_t dim, double init_value = 0.0)
		: position(dim, init_value), fitness(std::numeric_limits<double>::max()) {}
};

// SHADE class
class SHADE {
public:
	SHADE(size_t population_size, size_t dimension,
		const std::vector<double>& min_bound, const std::vector<double>& max_bound,
		size_t max_iter);
	void optimize(const std::function<double(const std::vector<double>&)>& loss_function);

	const Individual& getBestIndividual() const { return best_individual; }

private:
	size_t population_size;
	size_t dimension;
	std::vector<double> min_bound;
	std::vector<double> max_bound;
	size_t max_iter;

	std::vector<Individual> population;
	std::vector<double> memory_F;
	std::vector<double> memory_CR;
	Individual best_individual;

	void initializePopulation();
	double generateRandom(double min, double max);
	std::vector<double> mutate(const std::vector<double>& target, size_t target_idx);
	std::vector<double> crossover(const std::vector<double>& target, const std::vector<double>& donor, double CR);
};

#endif // SHADE_H
