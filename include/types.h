#pragma once

#include <Eigen/Core>
#include <vector>
#include <array>

namespace tdoa {

using Point = Eigen::Vector2d;
using Parameters = Eigen::VectorXd;

struct Device {
    char id;
    Point pos;
};

struct DistanceDiff {
    char source;
    char receiver1;
    char receiver2;
    double value;
};

struct InputData {
    std::array<Device, 3> sources;
    std::vector<DistanceDiff> dist_diffs;
};

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
    Eigen::VectorXd m;
    Eigen::VectorXd v;
    int t;
};

}
