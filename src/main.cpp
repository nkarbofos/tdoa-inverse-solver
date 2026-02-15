#include "solver.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

using namespace tdoa;

InputData generateTestData(bool add_noise = false, double noise_level = 0.01) {
    InputData data;
    
    data.sources[0] = {'D', Point(0.0, 0.0)};
    data.sources[1] = {'E', Point(10.0, 0.0)};
    data.sources[2] = {'F', Point(5.0, 5.0 * std::sqrt(3.0))};
    
    Point A_true(4.0, 3.0);
    Point B_true(-2.0, 5.0);
    Point C_true(1.0, -4.0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noise_level);
    
    for (const auto& source : data.sources) {
        const Point& S = source.pos;
        double AS = (A_true - source.pos).norm();
        double BS = (B_true - source.pos).norm();
        double CS = (C_true - source.pos).norm();
        
        double n_AB = add_noise ? noise(gen) : 0.0;
        double n_AC = add_noise ? noise(gen) : 0.0;
        double n_BC = add_noise ? noise(gen) : 0.0;
        
        data.dist_diffs.push_back({source.id, 'A', 'B', AS - BS + n_AB});
        data.dist_diffs.push_back({source.id, 'A', 'C', AS - CS + n_AC});
        data.dist_diffs.push_back({source.id, 'B', 'C', BS - CS + n_BC});
    }
    
    return data;
}

int main() {
    InputData data = generateTestData(false);
    TDoASolver solver(data);
    
    std::vector<Parameters> initial_guesses = {
        Eigen::VectorXd::Zero(6),
        (Eigen::VectorXd(6) << 5.0, 5.0, -5.0, 5.0, 0.0, -5.0).finished(),
        (Eigen::VectorXd(6) << 10.0, 10.0, -10.0, 10.0, 10.0, -10.0).finished()
    };
    
    SolverConfig config;
    config.verbose = true;
    config.print_every = 100;
    config.max_iterations = 2000;
    
    for (size_t i = 0; i < initial_guesses.size(); ++i) {
        std::cout << "\n\n=========Test " << i+1 << "=============\n";
        std::cout << "Initial: " << initial_guesses[i].transpose() << "\n";
        
        auto result = solver.solve(initial_guesses[i], config);
        if (result) {
            std::cout << "\nSolution found\n";
            std::cout << "A: " << (*result)[0] << " " << (*result)[1] << ")\n";
            std::cout << "B: " << (*result)[2] << " " << (*result)[3] << ")\n";
            std::cout << "C: " << (*result)[4] << " " << (*result)[5] << ")\n";
            std::cout << "Final loss: " << solver.computeLoss(*result) << "\n";
        } else {
            std::cout << "!!!!!!!!!!!Solution not found!!!!!!!!!!!\n";
        }
    }
    
    return 0;
}
