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
    throw std::runtime_error("dist_diff not found");

    return -1;
}

std::array<Point, 3> TDoASolver::computeReceivers(const Parameters& params) const {
    Point A(params[0], params[1]);
    Point B(params[2], params[3]);
    Point C(params[4], params[5]);

    return {A, B, C};
}

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
    
    Parameters grad = Eigen::VectorXd::Zero(6);
    for (const auto& source : data_.sources) {
        const Point& S = source.pos;
        double AS = distance(A, S);
        double BS = distance(B, S);
        double CS = distance(C, S);
        
        Eigen::Vector2d uAS, uBS, uCS;
        if (AS > 1e-12) {
            uAS = (A - S) / AS;
        } else {
            uAS = Eigen::Vector2d::Zero();
        }

        if (BS > 1e-12) {
            uBS = (B - S) / BS;
        } else {
            uBS = Eigen::Vector2d::Zero();
        }

        if (CS > 1e-12) {
            uCS = (C - S) / CS;
        } else {
            uCS = Eigen::Vector2d::Zero();
        }
        
        double AB_meas = findDistanceDiff(source.id, 'A', 'B');
        double AC_meas = findDistanceDiff(source.id, 'A', 'C');
        double BC_meas = findDistanceDiff(source.id, 'B', 'C');
        
        double AB_real = AS - BS;
        double AC_real = AS - CS;
        double BC_real = BS - CS;
        
        double AB_err = AB_real - AB_meas;
        double AC_err = AC_real - AC_meas;
        double BC_err = BC_real - BC_meas;
        
        Eigen::Vector2d grad_A = 2.0 * AB_err * uAS + 2.0 * AC_err * uAS;
        grad[0] += grad_A.x();
        grad[1] += grad_A.y();
        
        Eigen::Vector2d grad_B = -2.0 * AB_err * uBS + 2.0 * BC_err * uBS;
        grad[2] += grad_B.x();
        grad[3] += grad_B.y();
        
        Eigen::Vector2d grad_C = -2.0 * AC_err * uCS - 2.0 * BC_err * uCS;
        grad[4] += grad_C.x();
        grad[5] += grad_C.y();
    }
    
    return grad;
}

Parameters TDoASolver::computeNumericalGradient(const Parameters& params, double eps) const {
    double loss = computeLoss(params);
    Parameters grad = Eigen::VectorXd::Zero(6);
    for (int i = 0; i < 6; ++i) {
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

Parameters TDoASolver::adamStep(Parameters params, const Parameters& grad, AdamState& state, const AdamConfig& config) const {
    if (state.m.size() == 0) {
        state.m = Eigen::VectorXd::Zero(6);
        state.v = Eigen::VectorXd::Zero(6);
        state.t = 0;
    }

    ++state.t;
    state.m = config.beta1 * state.m + (1.0 - config.beta1) * grad;
    state.v = config.beta2 * state.v + (1.0 - config.beta2) * grad.cwiseProduct(grad);
    
    Eigen::VectorXd m_hat = state.m / (1.0 - std::pow(config.beta1, state.t));
    Eigen::VectorXd v_hat = state.v / (1.0 - std::pow(config.beta2, state.t));
    Eigen::VectorXd update = config.alpha * m_hat.array() / (v_hat.array().sqrt() + config.epsilon);
    
    return params - update;
}

std::optional<Parameters> TDoASolver::solve(const Parameters& initial, const SolverConfig& config) {
    Parameters params = initial;
    AdamState state;
    
    double prev_loss = computeLoss(params);
    double best_loss = prev_loss;
    Parameters best_params = params;
    int patience_cnt = 0;
    
    if (config.verbose) {
        bool grad_ok = checkGradient(params, 1e-3);
        std::cout << "grad_ok: " << grad_ok << "\n";
        std::cout << "prev_loss: " << prev_loss << "\n";
    }
    
    for (int iter = 0; iter < config.max_iterations; ++iter) {
        Parameters grad = computeGradient(params);
        params = adamStep(params, grad, state, config.adam);
        
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