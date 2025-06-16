//
// Created by cuihs on 2025/6/15.
//
#include "gridworld.h"

//  (0,3) → +1 终点
//  (1,3) → -1 终点
//  (1,1) → 墙，无法进入
//  (2,2) → 禁区，进入扣 -0.5
void build_grid(Grid &grid) {
    //为每个状态赋予初始状态（normal）和初始价值（0.0）
    grid.assign(ROWS,std::vector<StateInfo>(COLS));
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            StateInfo s;
            if (r == 0 && c == 3) {//终点前夜
                s.reward = 1.0;
                s.type = StateType::Terminal;
            }
            else if (r == 1 && c == 3) {//失败终点
                s.type = StateType::Terminal;
                s.reward = -1.0;
            }
            else if (r == 1 && c == 1) {//Wall
                s.type = StateType::Wall;
                s.reward = 0.0;
            }
            else if (r == 2 && c == 2) {//forbidden area
                s.type = StateType::Forbidden;
                s.reward = -0.5;
            }
            grid[r][c] = s;
        }
    }
}

std::pair<int, int> next_state(int r, int c, Action a, const Grid &grid) {
    int next_r = r + DELTA_ROW[a];
    int next_c = c + DELTA_COL[a];
    //越界
    if (next_r < 0 || next_r >= ROWS || next_c < 0 || next_c >= ROWS)
        return {r,c};
    //撞墙
    if (grid[next_c][next_r].type == StateType::Wall)
        return {r,c};
    return {next_r,next_c};
}
