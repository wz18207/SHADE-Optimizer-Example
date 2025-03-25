#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "SHADE.h"
#include "kmpe_cost_function.h"
#include "sphere_calibration.h"
#include "LevenbergMarquardt.h"

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

void print4x4Matrix(const Eigen::Matrix4d& matrix) {
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f |\n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f |\n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f |\n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

void checkDimensions(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
	if (v1.size() != v2.size()) {
		std::cerr << "Dimension mismatch: v1 size = " << v1.size() << ", v2 size = " << v2.size() << std::endl;
		exit(1);
	}
}

void splitPointClouds(const PointCloudT::Ptr& cloud_source, const PointCloudT::Ptr& cloud_target, DataK& data_k) {
	size_t num_points = cloud_source->size();
	assert(cloud_source->size() == cloud_target->size() && "PointCloud sizes do not match!");
	data_k.Points1.resize(3);
	data_k.Points2.resize(3);
	for (size_t i = 0; i < num_points; ++i) {
		data_k.Points1[0].conservativeResize(data_k.Points1[0].rows() + 1, 1);
		data_k.Points1[1].conservativeResize(data_k.Points1[1].rows() + 1, 1);
		data_k.Points1[2].conservativeResize(data_k.Points1[2].rows() + 1, 1);
		data_k.Points2[0].conservativeResize(data_k.Points2[0].rows() + 1, 1);
		data_k.Points2[1].conservativeResize(data_k.Points2[1].rows() + 1, 1);
		data_k.Points2[2].conservativeResize(data_k.Points2[2].rows() + 1, 1);
		data_k.Points1[0](data_k.Points1[0].rows() - 1) = cloud_source->points[i].x;
		data_k.Points1[1](data_k.Points1[1].rows() - 1) = cloud_source->points[i].y;
		data_k.Points1[2](data_k.Points1[2].rows() - 1) = cloud_source->points[i].z;
		data_k.Points2[0](data_k.Points2[0].rows() - 1) = cloud_target->points[i].x;
		data_k.Points2[1](data_k.Points2[1].rows() - 1) = cloud_target->points[i].y;
		data_k.Points2[2](data_k.Points2[2].rows() - 1) = cloud_target->points[i].z;

	}
	std::cout << "Points1 dimensions: x=" << data_k.Points1[0].rows()
		<< ", y=" << data_k.Points1[1].rows()
		<< ", z=" << data_k.Points1[2].rows() << std::endl;

	std::cout << "Points2 dimensions: x=" << data_k.Points2[0].rows()
		<< ", y=" << data_k.Points2[1].rows()
		<< ", z=" << data_k.Points2[2].rows() << std::endl;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " source.pcd target.pcd\n";
		return -1;
	}
	PointCloudT::Ptr cloud_source(new PointCloudT);
	PointCloudT::Ptr cloud_target(new PointCloudT);
	if (pcl::io::loadPCDFile(argv[1], *cloud_source) == -1 ||
		pcl::io::loadPCDFile(argv[2], *cloud_target) == -1) {
		PCL_ERROR("Error loading PCD files.\n");
		return -1;
	}
	std::cout << "Loaded source cloud: " << cloud_source->size() << " points.\n";
	std::cout << "Loaded target cloud: " << cloud_target->size() << " points.\n";
	DataK data_k;
	splitPointClouds(cloud_source, cloud_target, data_k);

	std::vector<double> fitness_5_sphere_min_bound = { 700, -2000, -2000, 700, -2000, -2000, 700, -2000, -2000, 700, -2000, -2000, 700, -2000, -2000, 700, -2000, -2000, 0 };
	std::vector<double> fitness_5_sphere_max_bound = { 4300, 2000, 2000, 4300, 2000, 2000, 4300, 2000, 2000, 4300, 2000, 2000, 4300, 2000, 2000, 4300, 2000, 2000, 200 };
	size_t fitness_5_sphere_dimension = 19;
	size_t fitness_5_population_size = 50;
	size_t fitness_5_max_iterations = 100;

	SHADE sphere_optimizer(fitness_5_population_size, fitness_5_sphere_dimension, fitness_5_sphere_min_bound, fitness_5_sphere_max_bound, fitness_5_max_iterations);
	auto fitness5_function = [&](const std::vector<double>& params) -> double {
		return Fitness5(params, data_k);  //Ft
	};
	sphere_optimizer.optimize(fitness5_function);
	const auto& best_sphere_solution = sphere_optimizer.getBestIndividual();
	std::vector<double> sphere_params = best_sphere_solution.position;
	// Assertion failed: aLhs.rows() == aRhs.rows() && aLhs.cols() == aRhs.cols(), 
	// file c:\program files\pcl 1.9.1\3rdparty\eigen\eigen3\eigen\src\core\cwisebinaryop.h, line 110
	// cout << "4" << endl;
	std::cout << "Optimized sphere parameters: ";
	for (const auto& p : sphere_params) std::cout << p << " ";
	std::cout << "\nSphere fitness: " << best_sphere_solution.fitness << "\n";

	LevenbergMarquardt lm;
	Eigen::VectorXd initial_guess = Eigen::Map<Eigen::VectorXd>(sphere_params.data(), sphere_params.size());
	Eigen::VectorXd target_value = Eigen::VectorXd::Zero(sphere_params.size());

	auto fitness5_residual_func = [&](Eigen::VectorXd params) -> Eigen::VectorXd {
		std::vector<double> params_vec(params.data(), params.data() + params.size());
		double fitness = Fitness5(params_vec, data_k);
		Eigen::VectorXd residual(1);
		residual[0] = fitness;
		return residual;
	};
	checkDimensions(initial_guess, target_value);
	// left.size() != dimensions || right.size() != dimensions
	// lest.size() = right.size() dimensions = 19 / 6 / ... need to fix the dimension error.
	Eigen::VectorXd optimized_sphere_params = lm(initial_guess, target_value, fitness5_residual_func);
	std::cout << "Optimized sphere parameters (Levenberg-Marquardt): ";
	for (int i = 0; i < optimized_sphere_params.size(); ++i) std::cout << optimized_sphere_params[i] << " ";
	std::cout << "\n";
	/*std::vector<double> lm_sphere_params;
	runLMOptimization(sphere_params, data_k, lm_sphere_params);

	std::cout << "LM Optimized sphere parameters: ";
	for (const auto& p : lm_sphere_params) std::cout << p << " ";
	std::cout << "\n";*/
	std::vector<double> fitness_3_pose_min_bound = { -M_PI, -M_PI / 2, -M_PI, -500, -500, -500 };
	std::vector<double> fitness_3_pose_max_bound = { M_PI, M_PI / 2, M_PI, 500, 500, 500 };
	size_t fitness_3_population_size = 50;
	size_t fitness_3_dimension = 6;
	size_t fitness_3_max_iterations = 100;
	SHADE pose_optimizer(fitness_3_population_size, fitness_3_dimension, fitness_3_pose_min_bound, fitness_3_pose_max_bound, fitness_3_max_iterations);
	KMPECostFunction kmpe_loss_function(0.1);
	auto kmpe_loss_function_wrapper = [&](const std::vector<double>& params) -> double {
		Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
		transform(0, 3) = params[0];
		transform(1, 3) = params[1];
		transform(2, 3) = params[2];
		Eigen::Matrix3d rotation;
		rotation = Eigen::AngleAxisd(params[5], Eigen::Vector3d::UnitZ()) *
			Eigen::AngleAxisd(params[4], Eigen::Vector3d::UnitY()) *
			Eigen::AngleAxisd(params[3], Eigen::Vector3d::UnitX());
		transform.block<3, 3>(0, 0) = rotation;
		PointCloudT::Ptr transformed_cloud(new PointCloudT);
		pcl::transformPointCloud(*cloud_source, *transformed_cloud, transform);
		return kmpe_loss_function.compute(transformed_cloud, cloud_target);
	};
	pose_optimizer.optimize(kmpe_loss_function_wrapper);
	const auto& best_pose_solution = pose_optimizer.getBestIndividual();
	std::vector<double> pose_params = best_pose_solution.position;
	std::cout << "Optimized pose parameters: ";
	for (const auto& p : pose_params) std::cout << p << " ";
	std::cout << "\nPose fitness: " << best_pose_solution.fitness << "\n";
	initial_guess = Eigen::Map<Eigen::VectorXd>(pose_params.data(), pose_params.size());
	Eigen::VectorXd optimized_pose_params = lm(initial_guess, target_value, [&](Eigen::VectorXd params) -> Eigen::VectorXd {
		std::vector<double> params_vec(params.data(), params.data() + params.size());
		Eigen::VectorXd residual(1);
		residual[0] = kmpe_loss_function_wrapper(params_vec);
		return residual;
		});
	std::cout << "Optimized pose parameters (Levenberg-Marquardt): ";
	for (int i = 0; i < optimized_pose_params.size(); ++i) std::cout << optimized_pose_params[i] << " ";
	std::cout << "\n";
	//// 使用 Levenberg-Marquardt 优化
	//std::vector<double> lm_pose_params;
	//runLMOptimization(pose_params, data_k, lm_pose_params);

	//std::cout << "LM Optimized pose parameters: ";
	//for (const auto& p : lm_pose_params) std::cout << p << " ";
	//std::cout << "\n";
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
	transformation_matrix(0, 3) = optimized_pose_params[0];
	transformation_matrix(1, 3) = optimized_pose_params[1];
	transformation_matrix(2, 3) = optimized_pose_params[2];
	Eigen::Matrix3d rotation;
	rotation = Eigen::AngleAxisd(optimized_pose_params[5], Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(optimized_pose_params[4], Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(optimized_pose_params[3], Eigen::Vector3d::UnitX());
	transformation_matrix.block<3, 3>(0, 0) = rotation;
	print4x4Matrix(transformation_matrix);
	PointCloudT::Ptr cloud_transformed(new PointCloudT);
	pcl::transformPointCloud(*cloud_source, *cloud_transformed, transformation_matrix);
	pcl::visualization::PCLVisualizer viewer("Demo");
	viewer.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<PointT>(cloud_target, 0, 255, 0), "target cloud");
	viewer.addPointCloud(cloud_transformed, pcl::visualization::PointCloudColorHandlerCustom<PointT>(cloud_transformed, 255, 0, 0), "transformed cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed cloud");
	while (!viewer.wasStopped()) {
		viewer.spinOnce();
	}
	return 0;
}
