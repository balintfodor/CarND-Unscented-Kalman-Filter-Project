#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse = VectorXd::Zero(4);
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		cerr << "Tools::CalculateRMSE: invalid input\n";
		return rmse;
	}

	for (size_t i = 0; i < estimations.size(); ++i) {
		VectorXd r = estimations[i] - ground_truth[i];
		r = r.array() * r.array();
		rmse += r;
	}
	rmse /= 1.0 * estimations.size();
	return rmse.array().sqrt();
}