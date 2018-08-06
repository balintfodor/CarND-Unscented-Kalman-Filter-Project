#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  n_x_ = 5;
  n_aug_ = 7;
  n_2aug1_ = 2 * n_aug_ + 1;
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(n_2aug1_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_2aug1_; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

MatrixXd UKF::generateAugmentedSigmaPoints(
    const VectorXd& x,
    const MatrixXd& P,
    double std_a,
    double std_yawdd,
    double lambda) const
{
    VectorXd x_aug = VectorXd::Zero(n_aug_);
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_2aug1_);
    
    x_aug.head(n_x_) = x;
    P_aug.topLeftCorner(n_x_, n_x_) = P;
    P_aug(5, 5) = std_a * std_a;
    P_aug(6, 6) = std_yawdd * std_yawdd;

    MatrixXd L = P_aug.llt().matrixL();

    Xsig_aug.col(0) = x_aug;
    double q = sqrt(lambda + n_aug_);
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + q * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - q * L.col(i);
    }
    return Xsig_aug;
}

void test::run() {
    testGenerateAugmentedSigmaPoints();
    testPredictSigmaPoints();
    testPredictMeanAndCovariance();
    testPredictRadarMeasurement();
    testPredictLidarMeasurement();
}

void test::testGenerateAugmentedSigmaPoints() {
    UKF ukf;
    MatrixXd Xsig_pred = ukf.generateAugmentedSigmaPoints(build::x(), build::P1(), 0.2, 0.2, 3 - ukf.n_aug_);

    MatrixXd Xsig_pred_exp = MatrixXd(ukf.n_aug_, 2 * ukf.n_aug_ + 1);
    Xsig_pred_exp <<
        5.7441,5.85768,5.7441,5.7441,5.7441,5.7441,5.7441,5.7441,5.63052,5.7441,5.7441,5.7441,5.7441,5.7441,5.7441,
        1.38,1.34566,1.52806,1.38,1.38,1.38,1.38,1.38,1.41434,1.23194,1.38,1.38,1.38,1.38,1.38,
        2.2049,2.28414,2.24557,2.29582,2.2049,2.2049,2.2049,2.2049,2.12566,2.16423,2.11398,2.2049,2.2049,2.2049,2.2049,
        0.5015,0.44339,0.631886,0.516923,0.595227,0.5015,0.5015,0.5015,0.55961,0.371114,0.486077,0.407773,0.5015,0.5015,0.5015,
        0.3528,0.299973,0.462123,0.376339,0.48417,0.418721,0.3528,0.3528,0.405627,0.243477,0.329261,0.22143,0.286879,0.3528,0.3528,
        0,0,0,0,0,0,0.34641,0,0,0,0,0,0,-0.34641,0,
        0,0,0,0,0,0,0,0.34641,0,0,0,0,0,0,-0.34641;

    assert(Xsig_pred.isApprox(Xsig_pred_exp, 10e-6));
}

void test::testPredictSigmaPoints() {

}

void test::testPredictMeanAndCovariance() {

}

void test::testPredictRadarMeasurement() {

}

void test::testPredictLidarMeasurement() {

}

void test::testUpdateState() {

}

MatrixXd test::build::Xsig_pred() {
    int n_x = 5;
    int n_aug = 7;
    MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
    Xsig_pred <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
    return Xsig_pred;
}

MatrixXd test::build::Xsig_aug() {
    int n_aug = 7;
    MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
    Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;
    return Xsig_aug;
}

VectorXd test::build::x() {
    VectorXd x = VectorXd(5);
    x <<
        5.7441,
        1.3800,
        2.2049,
        0.5015,
        0.3528;
    return x;
}

MatrixXd test::build::P1() {
    int n_x = 5;
    MatrixXd P = MatrixXd(n_x, n_x);
    P <<   0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
    return P;
}

MatrixXd test::build::P2() {
    int n_x = 5;
    MatrixXd P = MatrixXd(n_x, n_x);
    P <<
    0.0054342,  -0.002405,  0.0034157, -0.0034819, -0.00299378,
    -0.002405,    0.01084,   0.001492,  0.0098018,  0.00791091,
    0.0034157,   0.001492,  0.0058012, 0.00077863, 0.000792973,
    -0.0034819,  0.0098018, 0.00077863,   0.011923,   0.0112491,
    -0.0029937,  0.0079109, 0.00079297,   0.011249,   0.0126972;
    return P;
}

MatrixXd test::build::Zsig() {
    int n_aug = 7;
    int n_z = 3;
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
    Zsig <<
      6.1190,  6.2334,  6.1531,  6.1283,  6.1143,  6.1190,  6.1221,  6.1190,  6.0079,  6.0883,  6.1125,  6.1248,  6.1190,  6.1188,  6.12057,
     0.24428,  0.2337, 0.27316, 0.24616, 0.24846, 0.24428, 0.24530, 0.24428, 0.25700, 0.21692, 0.24433, 0.24193, 0.24428, 0.24515, 0.245239,
      2.1104,  2.2188,  2.0639,   2.187,  2.0341,  2.1061,  2.1450,  2.1092,  2.0016,   2.129,  2.0346,  2.1651,  2.1145,  2.0786,  2.11295;

}

VectorXd test::build::z_pred() {
    int n_z = 3;
    VectorXd z_pred = VectorXd(n_z);
    z_pred <<
      6.12155,
     0.245993,
      2.10313;
    return z_pred;
}

MatrixXd test::build::S() {
    int n_z = 3;
    MatrixXd S = MatrixXd(n_z, n_z);
    S <<
      0.0946171, -0.000139448,   0.00407016,
   -0.000139448,  0.000617548, -0.000770652,
     0.00407016, -0.000770652,    0.0180917;
    return S;
}

VectorXd test::build::z() {
    int n_z = 3;
    VectorXd z = VectorXd(n_z);
    z <<
      5.9214,   //rho in m
      0.2187,   //phi in rad
      2.0062;   //rho_dot in m/s
    return z;
}
