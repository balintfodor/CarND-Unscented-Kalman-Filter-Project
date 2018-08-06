#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

double angleNormalize(double a) {
    double z = std::remainder(a, 2.0 * M_PI);
    cout << a << " --> " << z << endl;
    return z;
}

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = false;

    n_x_ = 5;
    n_aug_ = 7;
    n_sig_ = 2 * n_aug_ + 1;
    x_ = VectorXd::Zero(n_x_);
    P_ = MatrixXd::Identity(n_x_, n_x_);
    weights_ = VectorXd(n_sig_);
    Xsig_pred_ = MatrixXd(n_x_, n_sig_);
    lambda_ = 3 - n_aug_;

    weights_(0) = lambda_ / (double)(lambda_ + n_aug_);
    for (int i = 1; i < n_sig_; ++i) {
        weights_(i) = 0.5 / (double)(lambda_ + n_aug_);
    }

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 30;

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

    test();
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
    is_initialized_= true;
    time_us_ = meas_package.timestamp_;
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
    Prediction(delta_t * 10e-6);
    time_us_ = meas_package.timestamp_;
    
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    } else {
        UpdateRadar(meas_package);
    }
}

MatrixXd UKF::GenerateSigmaPoints(double std_a, double std_yawdd) {

    VectorXd x_aug = VectorXd::Zero(n_aug_);
    x_aug.head(n_x_) = x_;

    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a * std_a;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd * std_yawdd;

    MatrixXd A = P_aug.llt().matrixL();
    double f = sqrt(lambda_ + n_aug_);

    MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) = x_aug + f * A.col(i);
        Xsig_aug.col(i + n_aug_ + 1) = x_aug - f * A.col(i);
    }
    return Xsig_aug;
}

void UKF::PredictSigmaPoints(MatrixXd Xsig_aug, double delta_t) {
    
    Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);
    double dt2 = delta_t * delta_t;
    for (int i = 0; i < n_sig_; i++) {
        double& px = Xsig_aug(0, i);
        double& py = Xsig_aug(1, i);
        double& v = Xsig_aug(2, i);
        double& yaw = Xsig_aug(3, i);
        double& yawd = Xsig_aug(4, i);
        double& nu_a = Xsig_aug(5, i);
        double& nu_yawdd = Xsig_aug(6, i);

        double& new_px = Xsig_pred_(0, i);
        double& new_py = Xsig_pred_(1, i);
        double& new_v = Xsig_pred_(2, i);
        double& new_yaw = Xsig_pred_(3, i);
        double& new_yawd = Xsig_pred_(4, i);

        double cos_yaw = cos(yaw);
        double sin_yaw = sin(yaw);

        double px_noise = 0.5 * dt2 * cos_yaw * nu_a;
        double py_noise = 0.5 * dt2 * sin_yaw * nu_a;
        double v_noise = delta_t * nu_a;
        double yaw_noise = 0.5 * dt2 * nu_yawdd;
        double yawd_noise = delta_t * nu_yawdd;

        if (fabs(yawd) > 10e-5) {
            double r = v / yawd;
            double yd = yawd * delta_t;
            new_px = px + r * ( sin(yaw + yd) - sin_yaw) + px_noise;
            new_py = py + r * (-cos(yaw + yd) + cos_yaw) + py_noise;
            new_yaw = yaw + yd + yaw_noise;
        } else {
            new_px = px + v * cos_yaw * delta_t + px_noise;
            new_py = py + v * sin_yaw * delta_t + py_noise;
            new_yaw = yaw + 0 + yaw_noise;
        }

        new_v = v + 0 + v_noise;
        new_yawd = yawd + 0 + yawd_noise;
        new_yaw = angleNormalize(new_yaw);
    }
}

void UKF::PredictMeanAndCovariance() {

    x_ = VectorXd::Zero(n_x_);
    P_ = MatrixXd::Zero(n_x_, n_x_);
    for (int i = 0; i < n_sig_; ++i) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    for (int i = 0; i < n_sig_; ++i) {
        VectorXd diff = Xsig_pred_.col(i) - x_;
        diff(3) = angleNormalize(diff(3));
        P_ += weights_(i) * diff * diff.transpose();
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    MatrixXd Xsig_aug = GenerateSigmaPoints(std_a_, std_yawdd_);
    PredictSigmaPoints(Xsig_aug, delta_t);
    PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    int n_z = 2;
    // 1.) predict measurement
    // 1.1) transform sigma points into measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_sig_);
    for (int i = 0; i < n_sig_; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    // 1.2) mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // 1.3) innovation covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < n_sig_; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    // 1.4) add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R <<    std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;
    S = S + R;

    // 2.) update state
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    // 2.1) calculate cross correlation matrix
    for (int i = 0; i < n_sig_; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = angleNormalize(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z = meas_package.raw_measurements_.head(n_z);
    VectorXd z_diff = z - z_pred;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    double nis = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
    std::cout << "LIDAR NIS: " << nis << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    // cout << "-> UKF::UpdateRadar\n";
    int n_z = 3;
    // 1.) predict measurement
    // 1.1) transform sigma points into measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_sig_);
    for (int i = 0; i < n_sig_; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double phi = Xsig_pred_(3, i);
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
        Zsig(1, i) = atan2(p_y, p_x);
        Zsig(2, i) = (p_x * cos(phi) + p_y * sin(phi)) * v / Zsig(0, i);
    }
    cout << "Zsig\n" << Zsig << "\n";
    // 1.2) mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    // cout << "z_pred\n" << z_pred << "\n";
    // 1.3) innovation covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    // cout << "S\n" << S << "\n";
    for (int i = 0; i < n_sig_; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        angleNormalize(z_diff(1));
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    // cout << "S\n" << S << "\n";
    // 1.4) add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R <<    std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;
    S = S + R;
    cout << "R\n" << R << "\n";
    cout << "S + R\n" << S << "\n";
    cout << "Zsig\n" << Zsig << "\n";
    cout << "z_pred\n" << z_pred << "\n";
    cout << "Xsig_pred_\n" << Xsig_pred_ << "\n";
    cout << "x_\n" << x_ << "\n";

    // 2.) update state
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
    // 2.1) calculate cross correlation matrix
    for (int i = 0; i < n_sig_; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        angleNormalize(z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        angleNormalize(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    cout << "Tc\n" << Tc << "\n";
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    cout << "K\n" << K << "\n";

    //residual
    VectorXd z = meas_package.raw_measurements_.head(n_z);
    // cout << "z\n" << z << "\n";
    // cout << "z_pred\n" << z_pred << "\n";
    VectorXd z_diff = z - z_pred;
    //angle normalization
    angleNormalize(z_diff(1));
    //update state mean and covariance matrix
    // cout << "z_diff\n" << z_diff << "\n";
    x_ = x_ + K * z_diff;
    angleNormalize(x_(3));
    P_ = P_ - K * S * K.transpose();

    // double nis = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
    // std::cout << "RADAR NIS: " << nis << std::endl;
    // cout << "x_\n" << x_ << "\nP_\n" << P_ << "\n---\n";
    // cout << "<- UKF::UpdateRadar\n";
}

void UKF::test() {

    testGenerateSigmaPoints();
    testPredictSigmaPoints();
    testPredictMeanAndCovariance();
}

void UKF::testGenerateSigmaPoints() {
    x_ <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;
    P_ <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    MatrixXd Xsig_aug = GenerateSigmaPoints(0.2, 0.2);

    MatrixXd Xsig_aug_exp(7, 15);
    Xsig_aug_exp <<
        5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,  5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
        1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,  1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
        2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,  2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
        0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,  0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
        0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528, 0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
            0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641,        0,
            0,        0,        0,        0,        0,        0,        0,  0.34641,        0,        0,        0,        0,        0,        0, -0.34641;
    
    assert(Xsig_aug_exp.isApprox(Xsig_aug, 10e-6));
}

void UKF::testPredictSigmaPoints() {

}

void UKF::testPredictMeanAndCovariance() {

}