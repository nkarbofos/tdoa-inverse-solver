#include "solver.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <map>

namespace tdoa {

TDoASolver::TDoASolver(const InputData& data) : data_(data) {}

double TDoASolver::distance(const Point& p1, const Point& p2) const {
    return (p1 - p2).norm();
}

double TDoASolver::findDistanceDiff(char source, char r1, char r2) const {
    for (const auto& m : data_.dist_diffs) {
        if (m.source == source && 
            ((m.receiver1 == r1 && m.receiver2 == r2) || 
             (m.receiver1 == r2 && m.receiver2 == r1))) {
            return m.value;
        }
    }

    std::cerr << "Error: " << source << " " << r1 << "-" << r2 << " not found\n";
    return -1;
}

std::array<Point, 3> TDoASolver::computeReceivers(const Parameters& params) const {
    double r_A = params[0];
    double phi_A = params[1];
    double phi_B = params[2];
    double phi_C = params[3];
    
    const Point& D = data_.sources[0].pos;

    double AD = r_A;
    double CD = AD - findDistanceDiff('D', 'A', 'C');
    double BD = AD - findDistanceDiff('D', 'A', 'B');
    
    Point A = Point(D.x() + AD * std::cos(phi_A),
                    D.y() + AD * std::sin(phi_A));
    Point B = Point(D.x() + BD * std::cos(phi_B),
                    D.y() + BD * std::sin(phi_B));
    Point C = Point(D.x() + CD * std::cos(phi_C),
                    D.y() + CD * std::sin(phi_C));

    return {A, B, C};
}

/*std::array<int, 3> TDoASolver::computeErrors(const std::array<Point, 3> receivers, const Device& source) const {
    const Point& S = source.pos;
    double AS = distance(A, S);
    double BS = distance(B, S);
    double CS = distance(C, S);
    
    Eigen::Vector2d uAS = (A - S) / AS;
    Eigen::Vector2d uBS = (B - S) / BS;
    Eigen::Vector2d uCS = (C - S) / CS;

    double AB_real = AS - BS;
    double AC_real = AS - CS;
    double BC_real = BS - CS;
    
    double AB_meas = findDistanceDiff(source.id, 'A', 'B');
    double AC_meas = findDistanceDiff(source.id, 'A', 'C');
    double BC_meas = findDistanceDiff(source.id, 'B', 'C');
    
    double AB_err = AB_real - AB_meas;
    double AC_err = AC_real - AC_meas;
    double BC_err = BC_real - BC_meas;
}*/

double TDoASolver::computeLoss(const Parameters& params) const {
    auto receivers = computeReceivers(params);
    const Point& A = receivers[0];
    const Point& B = receivers[1];
    const Point& C = receivers[2];
    
    double loss = 0.0;
    for (const auto& source : data_.sources) {
        const Point& S = source.pos;
        double AS = distance(A, S);
        double BS = distance(B, S);
        double CS = distance(C, S);
        
        double AB_real = AS - BS;
        double AC_real = AS - CS;
        double BC_real = BS - CS;
        
        double AB_meas = findDistanceDiff(source.id, 'A', 'B');
        double AC_meas = findDistanceDiff(source.id, 'A', 'C');
        double BC_meas = findDistanceDiff(source.id, 'B', 'C');
        
        loss += pow(AB_real - AB_meas, 2);
        loss += pow(AC_real - AC_meas, 2);
        loss += pow(BC_real - BC_meas, 2);
    }
    
    return loss;
}

Parameters TDoASolver::computeGradient(const Parameters& params) const {
    auto receivers = computeReceivers(params);
    const Point& A = receivers[0];
    const Point& B = receivers[1];
    const Point& C = receivers[2];
    
    const Point& D = data_.sources[0].pos;
    
    double AD = distance(A, D);
    double BD = distance(B, D);
    double CD = distance(C, D);

    double r_A = params[0];
    double phi_A = params[1];
    double phi_B = params[2];
    double phi_C = params[3];

    Eigen::Vector2d u_phi_A(-std::sin(phi_A), std::cos(phi_A));
    Eigen::Vector2d u_phi_B(-std::sin(phi_B), std::cos(phi_B));
    Eigen::Vector2d u_phi_C(-std::sin(phi_C), std::cos(phi_C));
    
    Parameters grad = Parameters::Zero();
    for (const auto& source : data_.sources) {
        if(source.id == 'D') {
            continue;
        }

        const Point& S = source.pos;
        double AS = distance(A, S);
        double BS = distance(B, S);
        double CS = distance(C, S);
        
        Eigen::Vector2d uAS = (A - S) / AS;
        Eigen::Vector2d uBS = (B - S) / BS;
        Eigen::Vector2d uCS = (C - S) / CS;
        
        double AB_meas = findDistanceDiff(source.id, 'A', 'B');
        double AC_meas = findDistanceDiff(source.id, 'A', 'C');
        double BC_meas = findDistanceDiff(source.id, 'B', 'C');
        
        double AB_real = AS - BS;
        double AC_real = AS - CS;
        double BC_real = BS - CS;
        
        double AB_err = AB_real - AB_meas;
        double AC_err = AC_real - AC_meas;
        double BC_err = BC_real - BC_meas;
        
        double dr_A = 0;
        dr_A += 2.0 * AB_err * (uAS.dot(u_phi_A) - uBS.dot(u_phi_B));
        dr_A += 2.0 * AC_err * (uAS.dot(u_phi_A) - uCS.dot(u_phi_C));
        dr_A += 2.0 * BC_err * (uBS.dot(u_phi_B) - uCS.dot(u_phi_C));

        double dphi_A = 2.0 * AD * (AB_err * uAS.dot(u_phi_A) + AC_err * uAS.dot(u_phi_A));
        double dphi_B = 2.0 * BD * (AB_err * (-uBS.dot(u_phi_B)) + BC_err * uBS.dot(u_phi_B));
        double dphi_C = 2.0 * CD * (AC_err * (-uCS.dot(u_phi_C)) + BC_err * (-uCS.dot(u_phi_C)));
        
        grad[0] += dr_A;
        grad[1] += dphi_A;
        grad[2] += dphi_B;
        grad[3] += dphi_C;
    }
    
    return grad;
}

Parameters TDoASolver::computeNumericalGradient(const Parameters& params, double eps) const {
    double loss = computeLoss(params);
    Parameters grad = Parameters::Zero();
    for (int i = 0; i < 4; ++i) {
        Parameters params_plus = params;
        params_plus[i] += eps;
        double loss_plus = computeLoss(params_plus);
        grad[i] = (loss_plus - loss) / eps;
    }
    
    return grad;
}

bool TDoASolver::checkGradient(const Parameters& params, double tolerance) const {
    Parameters grad_analytic = computeGradient(params);
    Parameters grad_numeric = computeNumericalGradient(params);
    double norm = std::max(grad_analytic.norm(), grad_numeric.norm());
    double diff = (grad_analytic - grad_numeric).norm();

    return (norm < 1e-12 || diff / norm < tolerance);
}

Parameters TDoASolver::adamStep(Parameters params, const Parameters& grad, 
                                AdamState& state, const AdamConfig& config) {
    ++state.t;
    state.m = config.beta1 * state.m + (1.0 - config.beta1) * grad;
    state.v = config.beta2 * state.v + (1.0 - config.beta2) * grad.cwiseProduct(grad);
    
    Eigen::Vector4d m_hat = state.m / (1.0 - std::pow(config.beta1, state.t));
    Eigen::Vector4d v_hat = state.v / (1.0 - std::pow(config.beta2, state.t));
    Eigen::Vector4d update = config.alpha * m_hat.array() / (v_hat.array().sqrt() + config.epsilon);
    
    return params - update;
}

std::optional<Parameters> TDoASolver::solve(const Parameters& initial, 
                                           const SolverConfig& config) {
    Parameters params = initial;
    AdamState state{Eigen::Vector4d::Zero(), Eigen::Vector4d::Zero(), 0};
    
    double prev_loss = computeLoss(params);
    double best_loss = prev_loss;
    Parameters best_params = params;
    int patience_cnt = 0;
    
    if (config.verbose) {
        bool grad_ok = checkGradient(params);
        std::cout << "grad_ok: " << grad_ok << "\n";
        std::cout << "prev_loss: " << prev_loss << "\n";
    }
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        Parameters grad = computeGradient(params);
        params = adamStep(params, grad, state, config.adam);
        
        for (int i = 1; i < 4; ++i) {
            params[i] = std::fmod(params[i], 2*M_PI);
            if (params[i] < 0) {
                params[i] += 2*M_PI;
            }
        }
        
        double loss = computeLoss(params);
        if (best_loss - loss > 1e-6) {
            best_loss = loss;
            best_params = params;
            patience_cnt = 0;
        } else {
            ++patience_cnt;
        }
        
        double grad_norm = grad.norm();
        bool _converged = grad_norm < config.grad_tolerance;
        bool _loss_stable = std::abs(prev_loss - loss) < config.tolerance;
        bool _early_stop = patience_cnt > config.patience;
        
        if (config.verbose && iter % config.print_every == 0) {
            std::cout << "Iter " << std::setw(4) << iter 
                      << " loss: " << std::scientific << std::setprecision(4) << loss
                      << " |grad|: " << grad_norm
                      << " delta loss: " << prev_loss - loss
                      << "\n";
        }
        
        if (_converged || _loss_stable || _early_stop) {
            if (config.verbose) {
                std::cout << "converged: " << _converged << "loss_stable: " << _loss_stable << "early_stop" << _early_stop << "\n";
            }
            return best_params;
        }
        
        prev_loss = loss;
    }
    
    if (config.verbose) {
        std::cout << "Max iterations reached\n";
    }
    return best_params;
}

}