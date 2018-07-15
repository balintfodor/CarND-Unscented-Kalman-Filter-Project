#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void angleNormalize(double& a) {
    // a = fmod(a + M_PI, 2.0 * M_PI) - M_PI;
    while (a> M_PI) a-=2.*M_PI;
    while (a<-M_PI) a+=2.*M_PI;
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

    n_x_ = 5;
    n_aug_ = 7;
    n_sig_ = 2 * n_aug_ + 1;
    x_ = VectorXd::Zero(n_x_);
    P_ = MatrixXd::Identity(n_x_, n_x_);
    weights_ = VectorXd(n_sig_);
    Xsig_pred_ = MatrixXd(n_x_, n_sig_);
    lambda_ = 3 - n_aug_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 3;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 3;

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
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

    cout << "-> UKF::ProcessMeasurement\n";
    switch (meas_package.sensor_type_) {
        case MeasurementPackage::LASER:
            cout << "=== LASER\n" << meas_package.timestamp_ << "\n" << meas_package.raw_measurements_ << "\n";
            if (use_laser_ == false) {
                return;
            }
            break;
        case MeasurementPackage::RADAR:
            cout << "=== RADAR\n" << meas_package.timestamp_ << "\n" << meas_package.raw_measurements_ << "\n";
            if (use_radar_ == false) {
                return;
            }
            break;
    }

    if (is_initialized_ == false) {
        switch (meas_package.sensor_type_) {
            case MeasurementPackage::LASER: {
                x_(0) = meas_package.raw_measurements_(0);
                x_(1) = meas_package.raw_measurements_(1);
                break;    
            }
            case MeasurementPackage::RADAR: {
                double r = meas_package.raw_measurements_(0);
                double phi = meas_package.raw_measurements_(1);
                x_(0) = r * cos(phi);
                x_(1) = r * sin(phi);
                break;
            }
        }
        is_initialized_= true;
        cout << "is_initialized_ == false\n";
        cout << "x_\n" << x_ << "\nP_\n" << P_ << "\ntime_us_\n" << time_us_ << "\n---\n";
        Prediction(0.0);
    } else {
        double delta_t = meas_package.timestamp_ - time_us_;
        Prediction(delta_t * 10e-6);
    }
    
    switch (meas_package.sensor_type_) {
        case MeasurementPackage::LASER: {
            UpdateLidar(meas_package);
            break;
        }
        case MeasurementPackage::RADAR: {
            UpdateRadar(meas_package);
            break;
        }
    }

    time_us_ = meas_package.timestamp_;

    cout << "is_initialized_ == false\n";
    cout << "x_\n" << x_ << "\nP_\n" << P_ << "\ntime_us_\n" << time_us_ << "\n---\n";

    cout << "<- UKF::ProcessMeasurement\n";
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    cout << "-> UKF::Prediction delta_t=" << delta_t << "\n";
    // 1) generate sigma points
    // 1.1) augment x vector
    VectorXd x_aug = VectorXd::Zero(n_aug_);
    x_aug.head(n_x_) = x_;

    // 1.2) augment covarainace
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

    // 1.3) create square root matrix
    MatrixXd A = P_aug.llt().matrixL();
    double f = sqrt(lambda_ + n_aug_);

    // 1.4) create augmented sigma points
    MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; ++i) {
        Xsig_aug.col(i + 1) = x_aug + f * A.col(i);
        Xsig_aug.col(i + n_aug_ + 1) = x_aug - f * A.col(i);
    }

    // 2) predict sigma points
    Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);
    for (int i = 0; i < n_sig_; i++) {
        // extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        // 2.1) predicted state values
        double px_p, py_p;
        double cos_yaw = cos(yaw);
        double sin_yaw = sin(yaw);
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        // 2.3) write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    cout << "Xsig_pred_\n" << Xsig_pred_ << "\n";

    // 3.) predict mean and covariance
    // 3.1) set weights
    weights_(0) = lambda_ / (double)(lambda_ + n_aug_);
    for (int i = 1; i < n_sig_; ++i) {
        weights_(i) = 0.5 / (double)(lambda_ + n_aug_);
    }

    // 3.2) predict state mean
    x_ = VectorXd::Zero(n_x_);
    P_ = MatrixXd::Zero(n_x_, n_x_);
    for (int i = 0; i < n_sig_; ++i) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }
    // 3.3) predict state covariance matrix
    for (int i = 0; i < n_sig_; ++i) {
        P_ += weights_(i) * (Xsig_pred_.col(i) - x_) * (Xsig_pred_.col(i) - x_).transpose();
    }
    cout << "x_\n" << x_ << "\nP_\n" << P_ << "\n---\n";
    cout << "<- UKF::Prediction\n";
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    std::cout << "lidar start\n";
    int n_z = 2;
    // 1.) predict measurement
    // 1.1) transform sigma points into measurement space
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    // 1.2) mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    // 1.3) innovation covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
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
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        angleNormalize(x_diff(3));

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
    cout << "-> UKF::UpdateRadar\n";
    int n_z = 3;
    // 1.) predict measurement
    // 1.1) transform sigma points into measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_sig_);
    for (int i = 0; i < n_sig_; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v  = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);
        Zsig(1, i) = atan2(p_y, p_x);
        Zsig(2, i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y);
    }
    cout << "Zsig\n" << Zsig << "\n";
    // 1.2) mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z);
    for (int i = 0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    cout << "z_pred\n" << z_pred << "\n";
    // 1.3) innovation covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z, n_z);
    cout << "S\n" << S << "\n";
    for (int i = 0; i < n_sig_; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        angleNormalize(z_diff(1));
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    cout << "S\n" << S << "\n";
    // 1.4) add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R <<    std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;
    S = S + R;
    cout << "S + R\n" << S << "\n";

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
    cout << "z\n" << z << "\n";
    cout << "z_pred\n" << z_pred << "\n";
    VectorXd z_diff = z - z_pred;
    //angle normalization
    angleNormalize(z_diff(1));
    //update state mean and covariance matrix
    cout << "z_diff\n" << z_diff << "\n";
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    double nis = (z - z_pred).transpose() * S.inverse() * (z - z_pred);
    std::cout << "RADAR NIS: " << nis << std::endl;
    cout << "x_\n" << x_ << "\nP_\n" << P_ << "\n---\n";
    cout << "<- UKF::UpdateRadar\n";
}
