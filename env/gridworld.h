//
// Created by cuihs on 2025/6/15.
//

#ifndef GRIDWORLD_H
#define GRIDWORLD_H
#include <vector>

constexpr int ROWS = 5;
constexpr int COLS = 5;

enum Action {UP = 0,RIGHT = 1,DOWN = 2,LEFT = 3,STAY = 4};
constexpr int ACTIONS = 5;
const int DELTA_ROW[ACTIONS] = {-1,0,1,0,0};//行偏移
const int DELTA_COL[ACTIONS] = {0,1,0,-1,0};//列偏移

//状态信息
enum class StateType {
    Normal,
    Terminal,
    Forbidden
};
struct StateInfo {
    StateType type = StateType::Normal;
    double reward = 0.0;
};

//网格
using Grid  = std::vector<std::vector<StateInfo>>;//类型别名，Grid是一个类型而非变量

//构建状态信息：定义奖励、终止态、墙
void build_grid(Grid& grid);

//下一个状态
std::pair<int,int> next_state(int r,int c,Action a,const Grid& grid);

#endif //GRIDWORLD_H
