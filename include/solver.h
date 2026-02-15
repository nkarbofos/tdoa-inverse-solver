#pragma once

#include "types.h"
#include <array>
#include <functional>
#include <optional>

namespace tdoa {
    struct AdamConfig {
        double alpha = 0.001;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
    };

    struct SolverConfig {
        AdamConfig adam;
        int max_iterations = 2000;
        double tolerance = 1e-8;
        double grad_tolerance = 1e-6;
        int patience = 50;
        bool verbose = true;
        int print_every = 100;
    };

    struct AdamState {
        Eigen::Vector4d m;
        Eigen::Vector4d v;
        int t;
    };

    class TDoASolver {
    public:
        explicit TDoASolver(const InputData& data);
        
        std::optional<Parameters> solve(const Parameters& initial_guess, 
                                        const SolverConfig& config = {});
        
        double computeLoss(const Parameters& params) const;
        
        Parameters computeGradient(const Parameters& params) const;
        
        Parameters computeNumericalGradient(const Parameters& params, 
                                            double eps = 1e-6) const;
        
        std::array<Point, 3> computeReceivers(const Parameters& params) const;

        const InputData& getData() const { return data_; }

    private:
        InputData data_;
        
        double distance(const Point& p1, const Point& p2) const;
        Point getReceiverPos(const Parameters& params, char receiver) const;
        double findDistanceDiff(char source, char r1, char r2) const;
        Parameters adamStep(Parameters params, const Parameters& grad, 
                            AdamState& state, const AdamConfig& config);
        bool checkGradient(const Parameters& params, double tolerance = 1e-4) const;
    };

}