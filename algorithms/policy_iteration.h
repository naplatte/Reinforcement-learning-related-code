//
// Created by cuihs on 2025/6/15.
//

#ifndef POLICY_ITERATION_H
#define POLICY_ITERATION_H
#include <vector>
#include "../env/gridworld.h"
#include "../env/mdp_config.h"
#include <cmath>

/*
 Algorithm idea:
 Given an initial policy (e.g., all actions are up), state values are initialized to 0 (the initial value doesn't matter, just define this variable with an initial value)
 Policy evaluation: Evaluate the state values for all (s,a) under such a policy. This process is actually solving the Bellman equation. We use iteration to solve it, so delta is introduced to determine whether v converges to the state value
 Policy improvement: Using the state values V obtained in the evaluation phase, calculate action values according to the Bellman formula (immediate reward + gamma * future return), select the action with the maximum action value, and update the policy
 */
void policy_iteration(const Grid& grid,std::vector<std::vector<double>>& V,std::vector<std::vector<int>>& policy) {
    //initialization
    V.assign(ROWS,std::vector<double>(COLS,0.0));
    policy.assign(ROWS,std::vector<int>(COLS,0));

    bool stable = false;//whether converged
    while (!stable) {
        //---policy evaluation---
        while (1) {
            double delta = 0.0;
            for (int r = 0; r < ROWS; ++r) {
                for (int c = 0; c < COLS; ++c) {
                    int a = policy[r][c];
                    auto [next_r,next_c] = next_state(r,c,static_cast<Action>(a),grid);
                    double new_val = grid[next_r][next_c].reward + GAMMA * V[next_r][next_c];
                    delta = std::max(delta,std::fabs(new_val - V[r][c]));
                    V[r][c] = new_val;
                }
            }
            //evaluation convergence: this is the iteration method to solve the Bellman equation, and V finally converges to the state value
            if (delta < THETA) break;
        }
        //---policy improvement---
        stable = true;
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                int old_a = policy[r][c];
                int best_a = old_a;
                double best_q = -1e9;
                for (int a = 0; a < ACTIONS; ++a) {
                    auto [next_r,next_c] = next_state(r,c,static_cast<Action>(a),grid);
                    double val = grid[next_r][next_c].reward + GAMMA * V[next_r][next_c];
                    if (val > best_q) {
                        best_q = val;
                        best_a = a;
                    }
                }
                policy[r][c] = best_a;
                if (best_a != old_a)
                    stable = false;
            }
        }
    }
}

#endif //POLICY_ITERATION_H
