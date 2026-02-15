#pragma once

#include <Eigen/Core>
#include <vector>
#include <array>

namespace tdoa {
    using Point = Eigen::Vector2d;
    using Parameters = Eigen::Vector4d;

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
}
