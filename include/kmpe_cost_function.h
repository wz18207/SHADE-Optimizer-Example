#ifndef KMPE_COST_FUNCTION_H
#define KMPE_COST_FUNCTION_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

class KMPECostFunction {
public:
    KMPECostFunction(double threshold) : threshold(threshold) {}

    double compute(const pcl::PointCloud<pcl::PointXYZ>::Ptr& predicted,
                   const pcl::PointCloud<pcl::PointXYZ>::Ptr& actual) const {
        if (predicted->size() != actual->size()) {
            throw std::runtime_error("Point clouds must have the same size for KMPE computation.");
        }

        double total_cost = 0.0;

        for (size_t i = 0; i < predicted->size(); ++i) {
            Eigen::Vector3d pred_point(predicted->points[i].x, predicted->points[i].y, predicted->points[i].z);
            Eigen::Vector3d actual_point(actual->points[i].x, actual->points[i].y, actual->points[i].z);

            double error = (pred_point - actual_point).norm();
            if (error <= threshold) {
                total_cost += 0.5 * error * error;
            } else {
                total_cost += threshold * (error - 0.5 * threshold);
            }
        }

        return total_cost;
    }

private:
    double threshold;
};

#endif // KMPE_cost_FUNCTION_H
