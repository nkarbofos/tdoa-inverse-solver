#include "solver.h"
#include <iostream>
#include <iomanip>

using namespace tdoa;

int main() {
    InputData data;
    data.sources[0] = {'D', Point(0, 0)};
    data.sources[1] = {'E', Point(10, 0)};
    data.sources[2] = {'F', Point(5, 8.66)};
    
    data.dist_diffs = {
        {'D', 'A', 'B', 2.0}, {'D', 'A', 'C', 3.0}, {'D', 'B', 'C', 1.0},
        {'E', 'A', 'B', 1.5}, {'E', 'A', 'C', 2.5}, {'E', 'B', 'C', 1.0},
        {'F', 'A', 'B', 2.2}, {'F', 'A', 'C', 3.2}, {'F', 'B', 'C', 1.0}
    };
    
    TDoASolver solver(data);
    std::vector<Parameters> test_points = {
        (Eigen::VectorXd(6) << 1.0, 1.0, -1.0, 2.0, 3.0, -2.0).finished(),
        (Eigen::VectorXd(6) << 5.0, 5.0, -5.0, 5.0, 0.0, -5.0).finished()
    };
    
    bool all_passed = true;
    for (int i = 0; i < test_points.size(); ++i) {
        std::cout << "Test point " << i+1 << ":\n";
        std::cout << "params: " << test_points[i].transpose() << "\n";
        
        bool grad_ok = solver.checkGradient(test_points[i], 1e-3);
        std::cout << "grad_ok: " << grad_ok << "\n";
        if (!grad_ok) {
            Parameters ga = solver.computeGradient(test_points[i]);
            Parameters gn = solver.computeNumericalGradient(test_points[i]);
            std::cout << "Analytic: " << ga.transpose() << "\n";
            std::cout << "Numeric: " << gn.transpose() << "\n";
            std::cout << "Diff: " << (ga - gn).transpose() << "\n";
        }
        
        all_passed = all_passed && grad_ok;
    }
    
    std::cout << "RESULT: " << (all_passed ? "Passed" : "FAIL") << "\n";    
    return !all_passed;
}
