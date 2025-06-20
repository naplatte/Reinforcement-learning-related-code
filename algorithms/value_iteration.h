//
// Created by cuihs on 2025/6/15.
//

#ifndef VALUE_ITERATION_H
#define VALUE_ITERATION_H
#include <complex>
#include <vector>
#include "../env/gridworld.h"
#include "../env/mdp_config.h"
/*
算法思路:
初始化状态值V（比如全设为0），定义一个策略并赋初值（赋予多少不重要，仅仅为定义变量赋初值）
使用贝尔曼公式反复更新每个状态的最大V,直到收敛，然后再从使用最大V求出最优动作（策略）
*/

//输入：环境grid,价值表v,最优策略policy
//policy[i][j]表示状态在(i,j)处的最优策略，使用上一轮迭代的v来计算本轮的最优策略
//V[i][j]表示状态在(i,j)处的状态值，拿本轮计算出的最优策略，来计算本轮的v
void value_iteration(const Grid& grid,std::vector<std::vector<double>>& V,std::vector<std::vector<int>>& policy) {
    //给一个V初值,用于迭代;policy是最后一次性提取出来的
    V.assign(ROWS,std::vector<double>(COLS,0.0));
    policy.assign(ROWS,std::vector<int>(COLS,-1));

    while (1) {
        double delta = 0.0;
        //遍历每一个(s,a)
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                double best_q = -1e9;//最大动作值
                for (int a = 0; a < ACTIONS; ++a) {
                    auto [next_r,next_c] = next_state(r,c,static_cast<Action> (a),grid);
                    double q_value = grid[next_r][next_c].reward + GAMMA * V[next_r][next_c];
                    if (q_value > best_q) best_q = q_value;
                }
                //值更新
                delta = std::max(delta,std::fabs(best_q - V[r][c]));//差值，看是否收敛
                V[r][c] = best_q;
            }
        }
        if (delta < THETA)  break;//收敛
    }

    //策略更新 - 一次性对每一个s更新策略（值收敛后，一次性提取最优策略）
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            double best_q = -1e9;
            int best_a = 0;
            for (int a = 0; a < ACTIONS; ++a) {
                auto [next_r,next_c] = next_state(r,c,static_cast<Action>(a),grid);
                double val = grid[r][c].reward + GAMMA * V[next_r][next_c];
                if (val > best_q) {
                    best_q = val;
                    best_a = a;
                }
            }
            policy[r][c] = best_a;
        }
    }

}

#endif //VALUE_ITERATION_H
