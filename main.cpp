#include <iostream>
#include <iomanip>
#include "env/gridworld.h"
#include "algorithms/policy_iteration.h"
#include "algorithms/value_iteration.h"
#include "algorithms/reinforce.h"
#include "algorithms/trpo.h"
#include "algorithms/ppo.h"

// Print grid representation of state values V
void print_grid(const std::vector<std::vector<double>> V) {
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << V[r][c] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Print policy
void print_policy(const std::vector<std::vector<int>>& policy, const Grid& grid) {
    const char arrows[] = "^>v<o";
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (grid[r][c].type == StateType::Terminal) {
                std::cout << arrows[policy[r][c]] << "(T) ";
            } else if (grid[r][c].type == StateType::Forbidden) {
                std::cout << arrows[policy[r][c]] << "(x) ";
            } else if (policy[r][c] >= 0 && policy[r][c] < ACTIONS) {
                std::cout << arrows[policy[r][c]] << " ";
            } else {
                std::cout << "? "; // uninitialized
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    Grid grid;
    build_grid(grid);

    std::vector<std::vector<double>> V;
    std::vector<std::vector<int>> policy;

    std::cout << "--- Value Iteration ---\n";
    value_iteration(grid, V, policy);
    print_grid(V);
    print_policy(policy, grid);

    std::cout << "--- Policy Iteration ---\n";
    policy_iteration(grid, V, policy);
    print_grid(V);
    print_policy(policy, grid);

    std::cout << "--- REINFORCE (Policy Gradient) ---\n";
    reinforce(grid, V, policy, 2000, 20, 0.01);  // 2000 episodes, update every 20 episodes, lr=0.01
    print_grid(V);
    print_policy(policy, grid);

    std::cout << "--- TRPO (Trust Region Policy Optimization) ---\n";
    trpo(grid, V, policy, 1500, 15, 0.01);  // 1500 episodes, update every 15 episodes, max_kl=0.01
    print_grid(V);
    print_policy(policy, grid);

    std::cout << "--- PPO (Proximal Policy Optimization) ---\n";
    ppo(grid, V, policy, 1500, 15, 0.001, 0.2);  // 1500 episodes, update every 15 episodes, lr=0.001, epsilon=0.2
    print_grid(V);
    print_policy(policy, grid);

    return 0;
}