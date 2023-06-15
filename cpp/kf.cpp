#include <kf.h>

KF::KF(float initial_pos, float initial_vel, float accel_variance):
    accel_variance{accel_variance} {
        covariance.setIdentity();
        state << initial_pos, initial_vel;
    }

KF::~KF(){}

void KF::predict(float dt){};
void KF::update(float measurement, float measurement_variance){};

const Eigen::Matrix<float,2,0> KF::get_state() const {
    return state;
}

const Eigen::Matrix<float,2,2> KF::get_cov() const {
    return covariance;
}

void KF::track_history(){
    this->_track_history = true;
}

const std::tuple<std::vector<Eigen::Matrix<float,2,2>>,std::vector<Eigen::Matrix<float,2,2>>> KF::get_history() const {
    return {x_history,p_history};
}