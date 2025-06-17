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
 算法思路：
 给定一个初始策略（比如action全是向上），状态值初值设为0（设为几不重要，仅仅是定义这个变量赋予一个初值）
 策略评估：评估在这样一个策略下，对于所有(s,a)的状态值，这个过程实际上就是求贝尔曼方程的过程，我们使用迭代法去求，所以引入delta就是为了判别v是否收敛为状态值
 策略改进：拿到在评估阶段求得的状态值V，根据贝尔曼公式去求动作值（即时奖励+gamma*未来回报），选择动作值最大的action，更新策略
 */
void policy_iteration(const Grid& grid,std::vector<std::vector<double>>& V,std::vector<std::vector<int>>& policy) {
    //初始化
    V.assign(ROWS,std::vector<double>(COLS,0.0));
    policy.assign(ROWS,std::vector<int>(COLS,0));

    bool stable = false;//是否收敛
    while (!stable) {
        //---策略评估---
        while (1) {
            double delta = 0.0;
            for (int r = 0; r < ROWS; ++r) {
                for (int c = 0; c < COLS; ++c) {
                    if (grid[r][c].type == StateType::Wall) continue;
                    int a = policy[r][c];
                    auto [next_r,next_c] = next_state(r,c,static_cast<Action>(a),grid);
                    double new_val = grid[next_r,next_c].reward + GAMMA * V[next_r,next_c];
                    delta = std::max(delta,std::fabs(new_val - V[r][c]));
                    V[r][c] = new_val;
                }
            }
            //评估收敛：这个就是迭代法求贝尔曼方程，最终V收敛到状态值
            if (delta < THETA) break;
        }
        //---策略改进---
        stable = true;
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                if (grid[r][c].type == StateType::Wall) continue;

                int old_a = policy[r][c];
                int best_a = old_a;
                double best_q = -1e9;
                for (int a = 0; a < ACTIONS; ++a) {
                    auto [next_r,next_c] = next_state(r,c,static_cast<Action>(a),grid);
                    double val = grid[next_r,next_c].reward + GAMMA * V[next_r,next_c];
                    if (val > best_q) {
                        best_q = val;
                        best_a = a;
                    }
                }
                policy[r,c] = best_a;
                if (best_a != old_a)
                    stable = false;
            }
        }
    }
}

#endif //POLICY_ITERATION_H
