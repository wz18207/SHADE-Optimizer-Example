#pragma once
#ifndef SPHERE_CALIBRATION_H
#define SPHERE_CALIBRATION_H
#include <numeric>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

struct DataK {
	std::vector<Eigen::MatrixXd> Points1;
	std::vector<Eigen::MatrixXd> Points2;
};

inline std::vector<double> lossless(const std::vector<double>& residuals, double sigma, double delta) {
	std::vector<double> L(residuals.size(), 0.0);
	for (size_t i = 0; i < residuals.size(); ++i) {
		double abs_residual = std::abs(residuals[i]);
		if (abs_residual <= delta) {
			L[i] = sigma * sigma * (std::sqrt(1 + std::pow(residuals[i] / sigma, 2)) - 1);
		}
		else if (abs_residual > delta && abs_residual <= delta * 1.5) {
			L[i] = delta * (abs_residual - 0.5 * delta) * (1 - (abs_residual - delta) / (0.5 * delta));
		}
		else {
			L[i] = delta * (abs_residual - 0.5 * delta);
		}
	}
	return L;
}

double Fitness5(const std::vector<double>& pos, const DataK& data_k) {
	double fitness = 0.0;
	double sigma0 = 2.0;
	double delta = 1.5;

	Eigen::MatrixXd Os(3, 6);
	for (int i = 0; i < 18; ++i) {
		Os(i / 6, i % 6) = pos[i];
	}
	double r = pos[18];
	/*
	Eigen::MatrixXd test = (data_k.Points1[0].colwise() - Os.col(2 * 0)).colwise().norm().array() - r;
	for (size_t k = 0; k < 3; ++k) {
		Eigen::MatrixXd diff1 = data_k.Points1[k].colwise() - Os.col(2 * k);
		Eigen::MatrixXd diff2 = data_k.Points2[k].colwise() - Os.col(2 * k + 1);
		Eigen::ArrayXd diff1_norm = diff1.colwise().norm().array() - r;
		Eigen::ArrayXd diff2_norm = diff2.colwise().norm().array() - r;
		std::vector<double> diff1_vec(diff1_norm.data(), diff1_norm.data() + diff1_norm.size());
		std::vector<double> diff2_vec(diff2_norm.data(), diff2_norm.data() + diff2_norm.size());
		auto loss1 = lossless(diff1_vec, sigma0, delta);
		auto loss2 = lossless(diff2_vec, sigma0, delta);
		fitness += std::accumulate(loss1.begin(), loss1.end(), 0.0);
		fitness += std::accumulate(loss2.begin(), loss2.end(), 0.0);
	}
	*/
	for (size_t k = 0; k < 3; ++k) {
		Eigen::MatrixXd diff1 = (data_k.Points1[k].topRows(3).colwise() - Os.col(2 * k)).colwise().norm().array() - r;
		Eigen::MatrixXd diff2 = (data_k.Points2[k].topRows(3).colwise() - Os.col(2 * k + 1)).colwise().norm().array() - r;
		std::vector<double> diff1_vec(diff1.data(), diff1.data() + diff1.size());
		std::vector<double> diff2_vec(diff2.data(), diff2.data() + diff2.size());
		auto loss1 = lossless(diff1_vec, sigma0, delta);
		auto loss2 = lossless(diff2_vec, sigma0, delta);
		fitness += std::accumulate(loss1.begin(), loss1.end(), 0.0);
		fitness += std::accumulate(loss2.begin(), loss2.end(), 0.0);
	}
	return fitness;
}

#endif  // SPHERE_CALIBRATION_H
