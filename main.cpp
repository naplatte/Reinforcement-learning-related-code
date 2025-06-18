#include <iostream>
#include <iomanip>
#include "env/gridworld.h"
#include "algorithms/policy_iteration.h"
#include "algorithms/value_iteration.h"


//打印状态值V的网格表示
void print_grid(const std::vector<std::vector<double>> V) {
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << V[r][c] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

//打印policy
void print_policy(const std::vector<std::vector<int>>& policy,const Grid& grid) {
    const char arrows[] = "↑→↓←";
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (grid[r][c].type == StateType::Terminal) std::cout << "T";
            else if (grid[r][c].type == StateType::Wall) std::cout << "#";
            else if (grid[r][c].type == StateType::Forbidden) std::cout << "x";
            else std::cout << arrows[policy[r][c]] << " ";
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

    return 0;
}