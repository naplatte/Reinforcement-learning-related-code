//
// Created by cuihs on 2025/6/15.
//
#include "gridworld.h"

void build_grid(Grid& grid) {
    grid.assign(ROWS, std::vector<StateInfo>(COLS));

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            StateInfo s;
            // 设置终点
            if (r == 4 && c == 4) {
                s.type = StateType::Terminal;
                s.reward = 1.0;
            }
            // 设置禁止区域
            else if ((r == 1 && c == 1) || (r == 2 && c == 3) || (r == 3 && c == 2)) {
                s.type = StateType::Forbidden;
                s.reward = -0.5;
            }
            // 普通状态默认 reward = 0.0，无需修改
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
    return {next_r,next_c};
}
