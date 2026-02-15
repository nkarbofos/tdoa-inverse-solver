#pragma once

#include "types.h"
#include <array>
#include <functional>
#include <optional>

namespace tdoa {

class TDoASolver {
public:
    explicit TDoASolver(const InputData& data);
    
    std::optional<Parameters> solve(const Parameters& initial, const SolverConfig& config = {});
    double computeLoss(const Parameters& params) const;
    Parameters computeGradient(const Parameters& params) const;
    Parameters computeNumericalGradient(const Parameters& params, double eps = 1e-6) const;
    std::array<Point, 3> computeReceivers(const Parameters& params) const;
    bool checkGradient(const Parameters& params, double tolerance = 1e-4) const;

    const InputData& getData() const { return data_; }

private:
    InputData data_;
    
    double distance(const Point& p1, const Point& p2) const;
    double findDistanceDiff(char source, char r1, char r2) const;
    Parameters adamStep(Parameters params, const Parameters& grad, AdamState& state, const AdamConfig& config) const;
};

}