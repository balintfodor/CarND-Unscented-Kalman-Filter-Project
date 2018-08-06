#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

class UKFDetails {
public:
  int n_x_;
  int n_aug_;
  int n_z_;
  int n_2aug1_;
  typedef std::pair<VectorXd, MatrixXd> MeanCovPair;

  UKFDetails(int n_x, int n_aug);
  MatrixXd generateAugmentedSigmaPoints(
    const VectorXd& x,
    const MatrixXd& P,
    double std_a,
    double std_yawdd,
    double lambda) const;
  MatrixXd predictSigmaPoints(
    const MatrixXd& Xsig_aug,
    double delta_t) const;
  MeanCovPair predictMeanAndCovariance(
    const MatrixXd& Xsig_pred,
    const VectorXd& weights) const;

  MeanCovPair predictRadarMeasurement(
    const MatrixXd& Xsig_pred,
    const VectorXd& weights,
    double std_radr,
    double std_radphi,
    double std_radrd) const;
  MeanCovPair predictLidarMeasurement(
    const MatrixXd& Xsig_pred,
    const VectorXd& weights,
    double std_laspx,
    double std_laspy) const;
  MeanCovPair updateRadarState(
    const MatrixXd& Xsig_pred,
    const MatrixXd& weights,
    const VectorXd& x,
    const MatrixXd& P,
    const MatrixXd& Zsig,
    const VectorXd& z_pred,
    const MatrixXd& S,
    const VectorXd& z) const;
  MeanCovPair updateLidarState(
    const MatrixXd& Xsig_pred,
    const MatrixXd& weights,
    const VectorXd& x,
    const MatrixXd& P,
    const MatrixXd& Zsig,
    const VectorXd& z_pred,
    const MatrixXd& S,
    const VectorXd& z) const;
};

namespace test {
  void run();

  namespace build {
    MatrixXd Xsig_pred();
    MatrixXd Xsig_aug();
    VectorXd x1();
    VectorXd x2();
    MatrixXd P1();
    MatrixXd P2();
    MatrixXd Zsig();
    VectorXd z_pred();
    MatrixXd S();
    VectorXd z();
  }

  void testGenerateAugmentedSigmaPoints();
  void testPredictSigmaPoints();
  void testPredictMeanAndCovariance();
  void testPredictRadarMeasurement();
  void testPredictLidarMeasurement();
  void testUpdateRadarState();
  void testUpdateLidarState();
}

#endif /* UKF_H */
