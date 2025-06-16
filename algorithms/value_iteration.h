//
// Created by cuihs on 2025/6/15.
//

#ifndef VALUE_ITERATION_H
#define VALUE_ITERATION_H
#include <vector>
#include "../env/gridworld.h"
#include "../env/mdp_config.h"

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
                if (grid[r][c] == StateType::Wall) continue;//跳过边界

            }
        }

    }

}

#endif //VALUE_ITERATION_H
