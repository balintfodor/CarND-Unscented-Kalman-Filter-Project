#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

double normalizeAngle(double rad) {
    return std::remainder(rad, 2.0 * M_PI);
}

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
  x_ = VectorXd::Zero(details_.n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(details_.n_x_, details_.n_x_) * 0.5;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
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
  
  n_x_ = details_.n_x_;
  n_aug_ = details_.n_aug_;
  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

UKF::~UKF() {}

void UKF::InitLaser(MeasurementPackage meas_package) {
    x_(0) = meas_package.raw_measurements_(0);
    x_(1) = meas_package.raw_measurements_(1);
}

void UKF::InitRadar(MeasurementPackage meas_package) {
    double r = meas_package.raw_measurements_(0);
    double phi = meas_package.raw_measurements_(1);
    x_(0) = r * cos(phi);
    x_(1) = r * sin(phi);
    x_(2) = meas_package.raw_measurements_(2);
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    if ((use_laser_ == false && meas_package.sensor_type_ == MeasurementPackage::LASER) || 
        (use_radar_ == false && meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
        return;
    }

    if (is_initialized_ == false) {
        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            InitLaser(meas_package);
        } else {
            InitRadar(meas_package);
        }
        is_initialized_= true;
        time_us_ = meas_package.timestamp_;
        return;
    }
    
    double delta_t = meas_package.timestamp_ - time_us_;
    Prediction(delta_t * 1e-6);
    time_us_ = meas_package.timestamp_;
    
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    } else {
        UpdateRadar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    MatrixXd Xsig_aug = details_.generateAugmentedSigmaPoints(x_, P_, std_a_, std_yawdd_, lambda_);
    Xsig_pred_ = details_.predictSigmaPoints(Xsig_aug, delta_t);
    UKFDetails::MeanCovPair xP = details_.predictMeanAndCovariance(Xsig_pred_, weights_);
    x_ = xP.first;
    P_ = xP.second;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    std::tuple<VectorXd, MatrixXd, MatrixXd> z_pred_S_Zsig =
        details_.predictLidarMeasurement(Xsig_pred_, weights_, std_laspx_, std_laspy_);

    VectorXd& z_pred = get<0>(z_pred_S_Zsig);
    MatrixXd& S = get<1>(z_pred_S_Zsig);
    MatrixXd& Zsig = get<2>(z_pred_S_Zsig);
    VectorXd& z = meas_package.raw_measurements_;

    UKFDetails::MeanCovPair xP = details_.updateLidarState(
        Xsig_pred_, weights_, x_, P_, Zsig, z_pred, S, z);
    x_ = xP.first;
    P_ = xP.second;

    double nis = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
    if (nis > 5.99) {
        above_nis_line += 1;
    }
    n_meas += 1;
    cout << (double)above_nis_line / (double)n_meas << endl;

    // std::cout << "LIDAR NIS (5.99): \t\t\t" << nis << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    
    std::tuple<VectorXd, MatrixXd, MatrixXd> z_pred_S_Zsig =
        details_.predictRadarMeasurement(Xsig_pred_, weights_, std_radr_, std_radphi_, std_radrd_);

    VectorXd& z_pred = get<0>(z_pred_S_Zsig);
    MatrixXd& S = get<1>(z_pred_S_Zsig);
    MatrixXd& Zsig = get<2>(z_pred_S_Zsig);
    VectorXd& z = meas_package.raw_measurements_;

    UKFDetails::MeanCovPair xP = details_.updateRadarState(
        Xsig_pred_, weights_, x_, P_, Zsig, z_pred, S, z);
    x_ = move(xP.first);
    P_ = move(xP.second);

    double nis = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
    if (nis > 7.81) {
        above_nis_line += 1;
    }
    n_meas += 1;
    cout << (double)above_nis_line / (double)n_meas << endl;

    // std::cout << "RADAR NIS (7.81): " << nis << std::endl;
}

UKFDetails::UKFDetails() : n_x_(5), n_aug_(7) {
    n_2aug1_ = 2 * n_aug_ + 1;
}

MatrixXd UKFDetails::generateAugmentedSigmaPoints(
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

MatrixXd UKFDetails::predictSigmaPoints(
    const MatrixXd& Xsig_aug,
    double delta_t) const
{
    MatrixXd Xsig_pred = MatrixXd::Zero(n_x_, n_2aug1_);
    double dt2 = delta_t * delta_t;
    for (int i = 0; i < n_2aug1_; i++) {
        const double& px = Xsig_aug(0, i);
        const double& py = Xsig_aug(1, i);
        const double& v = Xsig_aug(2, i);
        const double& yaw = Xsig_aug(3, i);
        const double& yawd = Xsig_aug(4, i);
        const double& nu_a = Xsig_aug(5, i);
        const double& nu_yawdd = Xsig_aug(6, i);

        double& new_px = Xsig_pred(0, i);
        double& new_py = Xsig_pred(1, i);
        double& new_v = Xsig_pred(2, i);
        double& new_yaw = Xsig_pred(3, i);
        double& new_yawd = Xsig_pred(4, i);

        double cos_yaw = cos(yaw);
        double sin_yaw = sin(yaw);

        double px_noise = 0.5 * dt2 * cos_yaw * nu_a;
        double py_noise = 0.5 * dt2 * sin_yaw * nu_a;
        double v_noise = delta_t * nu_a;
        double yaw_noise = 0.5 * dt2 * nu_yawdd;
        double yawd_noise = delta_t * nu_yawdd;

        if (fabs(yawd) > 1e-4) {
            double r = v / yawd;
            double yd = yawd * delta_t;
            new_px = px + r * ( sin(yaw + yd) - sin_yaw) + px_noise;
            new_py = py + r * (-cos(yaw + yd) + cos_yaw) + py_noise;
            new_yaw = yaw + yd + yaw_noise;
        } else {
            new_px = px + v * cos_yaw * delta_t + px_noise;
            new_py = py + v * sin_yaw * delta_t + py_noise;
            new_yaw = yaw + yaw_noise;
        }

        new_v = v + v_noise;
        new_yawd = yawd + yawd_noise;
    }
    return Xsig_pred;
}

UKFDetails::MeanCovPair UKFDetails::predictMeanAndCovariance(
    const MatrixXd& Xsig_pred,
    const VectorXd& weights) const
{
    VectorXd x = VectorXd::Zero(n_x_);
    MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
    for (int i = 0; i < n_2aug1_; ++i) {
        x += weights(i) * Xsig_pred.col(i);
    }

    for (int i = 0; i < n_2aug1_; ++i) {
        VectorXd diff = Xsig_pred.col(i) - x;
        diff(3) = normalizeAngle(diff(3));
        P += weights(i) * diff * diff.transpose();
    }
    return make_pair(x, P);
}

std::tuple<VectorXd, MatrixXd, MatrixXd> UKFDetails::predictRadarMeasurement(
    const MatrixXd& Xsig_pred,
    const VectorXd& weights,
    double std_radr,
    double std_radphi,
    double std_radrd) const
{
    static int n_z = 3;
    MatrixXd Zsig = MatrixXd(n_z, n_2aug1_);

    for (int i = 0; i < n_2aug1_; i++) {
        const double& p_x = Xsig_pred(0, i);
        const double& p_y = Xsig_pred(1, i);
        const double& v  = Xsig_pred(2, i);
        const double& yaw = Xsig_pred(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
        Zsig(1, i) = atan2(p_y, p_x);
        Zsig(2, i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y);
    }

    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < n_2aug1_; i++) {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }

    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < n_2aug1_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = normalizeAngle(z_diff(1));
        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_radr * std_radr, 0, 0,
        0, std_radphi * std_radphi, 0,
        0, 0, std_radrd * std_radrd;
    S = S + R;

    return std::tuple<VectorXd, MatrixXd, MatrixXd>(z_pred, S, Zsig);
}

std::tuple<VectorXd, MatrixXd, MatrixXd> UKFDetails::predictLidarMeasurement(
    const MatrixXd& Xsig_pred,
    const VectorXd& weights,
    double std_laspx,
    double std_laspy) const
{
    static int n_z = 2;
    MatrixXd Zsig = MatrixXd(n_z, n_2aug1_);

    for (int i = 0; i < n_2aug1_; i++) {
        Zsig(0, i) = Xsig_pred(0, i);
        Zsig(1, i) = Xsig_pred(1, i);
    }

    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < n_2aug1_; i++) {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }

    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < n_2aug1_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_laspx * std_laspx, 0,
        0, std_laspy * std_laspy;
    S = S + R;

    return std::tuple<VectorXd, MatrixXd, MatrixXd>(z_pred, S, Zsig);
}

UKFDetails::MeanCovPair UKFDetails::updateRadarState(
    const MatrixXd& Xsig_pred,
    const MatrixXd& weights,
    const VectorXd& x,
    const MatrixXd& P,
    const MatrixXd& Zsig,
    const VectorXd& z_pred,
    const MatrixXd& S,
    const VectorXd& z) const
{
    static int n_z = 3;
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < n_2aug1_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        z_diff(1) = normalizeAngle(z_diff(1));
        VectorXd x_diff = Xsig_pred.col(i) - x;
        x_diff(3) = normalizeAngle(x_diff(3));
        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }
    
    MatrixXd K = Tc * S.inverse();
    VectorXd z_diff = z - z_pred;

    z_diff(1) = normalizeAngle(z_diff(1));

    return make_pair(x + K * z_diff, P - K * S * K.transpose());
}

UKFDetails::MeanCovPair UKFDetails::updateLidarState(
    const MatrixXd& Xsig_pred,
    const MatrixXd& weights,
    const VectorXd& x,
    const MatrixXd& P,
    const MatrixXd& Zsig,
    const VectorXd& z_pred,
    const MatrixXd& S,
    const VectorXd& z) const
{
    static int n_z = 2;
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    for (int i = 0; i < n_2aug1_; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd x_diff = Xsig_pred.col(i) - x;
        x_diff(3) = normalizeAngle(x_diff(3));
        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }
    
    MatrixXd K = Tc * S.inverse();
    VectorXd z_diff = z - z_pred;

    return make_pair(x + K * z_diff, P - K * S * K.transpose());
}

void test::run() {
    testGenerateAugmentedSigmaPoints();
    testPredictSigmaPoints();
    testPredictMeanAndCovariance();
    testPredictRadarMeasurement();
    testUpdateRadarState();
}

void test::testGenerateAugmentedSigmaPoints() {
    UKFDetails ukfd;
    MatrixXd Xsig_aug = ukfd.generateAugmentedSigmaPoints(build::x1(), build::P1(), 0.2, 0.2, 3 - ukfd.n_aug_);

    MatrixXd Xsig_aug_exp = MatrixXd(ukfd.n_aug_, 2 * ukfd.n_aug_ + 1);
    Xsig_aug_exp <<
        5.7441,5.85768,5.7441,5.7441,5.7441,5.7441,5.7441,5.7441,5.63052,5.7441,5.7441,5.7441,5.7441,5.7441,5.7441,
        1.38,1.34566,1.52806,1.38,1.38,1.38,1.38,1.38,1.41434,1.23194,1.38,1.38,1.38,1.38,1.38,
        2.2049,2.28414,2.24557,2.29582,2.2049,2.2049,2.2049,2.2049,2.12566,2.16423,2.11398,2.2049,2.2049,2.2049,2.2049,
        0.5015,0.44339,0.631886,0.516923,0.595227,0.5015,0.5015,0.5015,0.55961,0.371114,0.486077,0.407773,0.5015,0.5015,0.5015,
        0.3528,0.299973,0.462123,0.376339,0.48417,0.418721,0.3528,0.3528,0.405627,0.243477,0.329261,0.22143,0.286879,0.3528,0.3528,
        0,0,0,0,0,0,0.34641,0,0,0,0,0,0,-0.34641,0,
        0,0,0,0,0,0,0,0.34641,0,0,0,0,0,0,-0.34641;

    assert(Xsig_aug.isApprox(Xsig_aug_exp, 1e-5));
    cout <<  __PRETTY_FUNCTION__ << " passed\n";
}

void test::testPredictSigmaPoints() {
    UKFDetails ukfd;
    MatrixXd Xsig_pred = ukfd.predictSigmaPoints(build::Xsig_aug(), 0.1);

    MatrixXd Xsig_pred_exp = MatrixXd(ukfd.n_x_, ukfd.n_2aug1_);
    Xsig_pred_exp <<
        5.93553,6.06251,5.92217,5.9415,5.92361,5.93516,5.93705,5.93553,5.80832,5.94481,5.92935,5.94553,5.93589,5.93401,5.93553,
        1.48939,1.44673,1.66484,1.49719,1.508,1.49001,1.49022,1.48939,1.5308,1.31287,1.48182,1.46967,1.48876,1.48855,1.48939,
        2.2049,2.28414,2.24557,2.29582,2.2049,2.2049,2.23954,2.2049,2.12566,2.16423,2.11398,2.2049,2.2049,2.17026,2.2049,
        0.53678,0.473387,0.678098,0.554557,0.643644,0.543372,0.53678,0.538512,0.600173,0.395462,0.519003,0.429916,0.530188,0.53678,0.535048,
        0.3528,0.299973,0.462123,0.376339,0.48417,0.418721,0.3528,0.387441,0.405627,0.243477,0.329261,0.22143,0.286879,0.3528,0.318159;

    assert(Xsig_pred.isApprox(Xsig_pred_exp, 1e-5));
    cout <<  __PRETTY_FUNCTION__ << " passed\n";
}

void test::testPredictMeanAndCovariance() {
    UKF ukf;
    UKFDetails ukfd;

    UKFDetails::MeanCovPair xP = ukfd.predictMeanAndCovariance(build::Xsig_pred(), ukf.weights_);
    
    VectorXd x_exp = VectorXd(ukfd.n_x_);
    x_exp << 5.93637, 1.49035, 2.20528, 0.536853, 0.353577;

    MatrixXd P_exp = MatrixXd(ukfd.n_x_, ukfd.n_x_);
    P_exp << 0.00543425,-0.0024053,0.00341576,-0.00348196,-0.00299378,
        -0.0024053,0.010845,0.0014923,0.00980182,0.00791091,
        0.00341576,0.0014923,0.00580129,0.000778632,0.000792973,
        -0.00348196,0.00980182,0.000778632,0.0119238,0.0112491,
        -0.00299378,0.00791091,0.000792973,0.0112491,0.0126972;

    assert(xP.first.isApprox(x_exp, 1e-5));
    assert(xP.second.isApprox(P_exp, 1e-5));

    cout <<  __PRETTY_FUNCTION__ << " passed\n";
}

void test::testPredictRadarMeasurement() {

    UKF ukf;
    UKFDetails ukfd;

    std::tuple<VectorXd, MatrixXd, MatrixXd> z_pred_S_Zsig = ukfd.predictRadarMeasurement(build::Xsig_pred(), ukf.weights_, 0.3, 0.0175, 0.1);

    VectorXd z_pred_exp = VectorXd(3);
    z_pred_exp << 6.12155,0.245993,2.10313;
    MatrixXd S_exp = MatrixXd(3, 3);
    S_exp << 0.0946171, -0.000139448, 0.00407016,
        -0.000139448, 0.000617548, -0.000770652,
        0.00407016, -0.000770652, 0.0180917;

    assert(get<0>(z_pred_S_Zsig).isApprox(z_pred_exp, 1e-5));
    assert(get<1>(z_pred_S_Zsig).isApprox(S_exp, 1e-5));

    cout <<  __PRETTY_FUNCTION__ << " passed\n";
}

void test::testUpdateRadarState() {

    UKF ukf;
    UKFDetails ukfd;

    UKFDetails::MeanCovPair xP = ukfd.updateRadarState(
        build::Xsig_pred(), ukf.weights_, build::x2(),
        build::P2(), build::Zsig(), build::z_pred(),
        build::S(), build::z());

    VectorXd x_exp = VectorXd(ukfd.n_x_);
    x_exp << 5.92276,1.41823,2.15593,0.489274,0.321338;

    MatrixXd P_exp = MatrixXd(ukfd.n_x_, ukfd.n_x_);
    P_exp << 
        0.00361579,-0.000357881,0.00208316,-0.000937196,-0.00071727,
        -0.000357881,0.00539867,0.00156846,0.00455342,0.00358885,
        0.00208316,0.00156846,0.00410651,0.00160333,0.00171811,
        -0.000937196,0.00455342,0.00160333,0.00652634,0.00669436,
        -0.00071719,0.00358884,0.00171811,0.00669426,0.00881797;

    assert(xP.first.isApprox(x_exp, 1e-5));
    assert(xP.second.isApprox(P_exp, 1e-5));

    cout <<  __PRETTY_FUNCTION__ << " passed\n";
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

VectorXd test::build::x2() {
    VectorXd x = VectorXd(5);
    x <<
        5.93637,
        1.49035,
        2.20528,
        0.536853,
        0.353577;
    return x;
}

VectorXd test::build::x1() {
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
    return Zsig;
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
      5.9214,
      0.2187,
      2.0062;
    return z;
}
