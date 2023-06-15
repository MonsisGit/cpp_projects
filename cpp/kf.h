#include <vector>
#include <Eigen/Dense>
#include <tuple>

class KF{
private:
    float accel_variance {0};
    Eigen::Matrix<float,2,0> state; 
    Eigen::Matrix<float,2,2> covariance;
    std::vector<Eigen::Matrix<float,2,2>> x_history;
    std::vector<Eigen::Matrix<float,2,2>> p_history;
    bool _track_history = false;
    
public:
    KF(float initial_pos, float initial_vel, float accel_variance);
    ~KF();
    void predict(float dt);
    void update(float measurement, float measurement_variance);
    const Eigen::Matrix<float,2,0> get_state() const;
    const Eigen::Matrix<float,2,2> get_cov() const;
    void track_history();
    const std::tuple<std::vector<Eigen::Matrix<float,2,2>>,std::vector<Eigen::Matrix<float,2,2>>> get_history() const;

    
};