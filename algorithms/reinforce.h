//
// Created by cuihs on 2025/6/15.
//

#ifndef REINFORCE_H
#define REINFORCE_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "../env/gridworld.h"
#include "../env/mdp_config.h"

// 经验回放缓冲区中的轨迹结构
struct Trajectory {
    std::vector<std::pair<int, int>> states;  // 状态序列 (r, c)
    std::vector<int> actions;                 // 动作序列
    std::vector<double> rewards;              // 奖励序列
    double total_return;                      // 整条轨迹的回报（折扣累加）
};

// 策略网络（简单的线性策略）
class PolicyNetwork {
private:
    std::vector<std::vector<std::vector<double>>> theta;  // 策略参数 [state_r][state_c][action]
    std::mt19937 rng;  // 随机数生成器
    
public:
    PolicyNetwork() : rng(std::random_device{}()) {
        // 初始化策略参数，每个状态-动作对的logit
        theta.assign(ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
    }
    
    // 获取动作概率分布
    std::vector<double> get_action_probs(int r, int c) {
        std::vector<double> logits = theta[r][c];
        std::vector<double> probs(ACTIONS);
        
        // 计算softmax
        double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (int a = 0; a < ACTIONS; ++a) {
            probs[a] = std::exp(logits[a] - max_logit);
            sum_exp += probs[a];
        }
        
        // 归一化
        for (int a = 0; a < ACTIONS; ++a) {
            probs[a] /= sum_exp;
        }
        
        return probs;
    }
    
    // 根据策略采样动作
    int sample_action(int r, int c) {
        std::vector<double> probs = get_action_probs(r, c);
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }
    
    // 获取动作概率
    double get_action_prob(int r, int c, int action) {
        std::vector<double> probs = get_action_probs(r, c);
        return probs[action];
    }
    
    // 更新策略参数
    void update_theta(const std::vector<Trajectory>& trajectories, double learning_rate) {
        // 计算每个状态-动作对的梯度
        std::vector<std::vector<std::vector<double>>> gradients(
            ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
        
        for (const auto& traj : trajectories) {
            for (size_t t = 0; t < traj.states.size(); ++t) {
                int r = traj.states[t].first;
                int c = traj.states[t].second;
                int action = traj.actions[t];
                
                // 计算策略梯度
                std::vector<double> probs = get_action_probs(r, c);
                for (int a = 0; a < ACTIONS; ++a) {
                    if (a == action) {
                        gradients[r][c][a] += (1.0 - probs[a]) * traj.total_return;
                    } else {
                        gradients[r][c][a] += (-probs[a]) * traj.total_return;
                    }
                }
            }
        }
        
        // 更新参数
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                for (int a = 0; a < ACTIONS; ++a) {
                    theta[r][c][a] += learning_rate * gradients[r][c][a];
                }
            }
        }
    }
    
    // 获取最优策略（选择概率最高的动作）
    std::vector<std::vector<int>> get_optimal_policy() {
        std::vector<std::vector<int>> policy(ROWS, std::vector<int>(COLS));
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                std::vector<double> probs = get_action_probs(r, c);
                policy[r][c] = std::max_element(probs.begin(), probs.end()) - probs.begin();
            }
        }
        return policy;
    }
};

// 运行一个episode并返回轨迹
Trajectory run_episode(const Grid& grid, PolicyNetwork& policy_net, int max_steps = 1000) {
    Trajectory traj;
    traj.total_return = 0.0;
    
    // 随机选择起始状态（避开禁止区域）
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_r(0, ROWS - 1);
    std::uniform_int_distribution<int> dist_c(0, COLS - 1);
    
    int r, c;
    do {
        r = dist_r(rng);
        c = dist_c(rng);
    } while (grid[r][c].type == StateType::Forbidden);
    
    double gamma_power = 1.0;  // gamma^t
    
    for (int step = 0; step < max_steps; ++step) {
        // 记录当前状态
        traj.states.emplace_back(r, c);
        
        // 选择动作
        int action = policy_net.sample_action(r, c);
        traj.actions.push_back(action);
        
        // 执行动作
        auto [next_r, next_c] = next_state(r, c, static_cast<Action>(action), grid);
        
        // 获取奖励
        double reward = grid[next_r][next_c].reward;
        traj.rewards.push_back(reward);
        
        // 累积折扣回报
        traj.total_return += gamma_power * reward;
        gamma_power *= GAMMA;
        
        // 检查是否到达终止状态
        if (grid[next_r][next_c].type == StateType::Terminal) {
            break;
        }
        
        // 检查是否进入禁止区域
        if (grid[next_r][next_c].type == StateType::Forbidden) {
            break;
        }
        
        r = next_r;
        c = next_c;
    }
    
    return traj;
}

// REINFORCE算法主函数
void reinforce(const Grid& grid, 
               std::vector<std::vector<double>>& V, 
               std::vector<std::vector<int>>& policy,
               int num_episodes = 1000,
               int episodes_per_update = 10,
               double learning_rate = 0.01) {
    
    PolicyNetwork policy_net;
    std::vector<Trajectory> trajectories;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        // 运行一个episode
        Trajectory traj = run_episode(grid, policy_net);
        trajectories.push_back(traj);
        
        // 每收集一定数量的episode就更新一次策略
        if ((episode + 1) % episodes_per_update == 0) {
            policy_net.update_theta(trajectories, learning_rate);
            trajectories.clear();  // 清空轨迹缓冲区
        }
    }
    
    // 获取最终的最优策略
    policy = policy_net.get_optimal_policy();
    
    // 计算状态值函数（可选，用于显示）
    V.assign(ROWS, std::vector<double>(COLS, 0.0));
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (grid[r][c].type == StateType::Terminal) {
                V[r][c] = grid[r][c].reward;
            } else if (grid[r][c].type == StateType::Forbidden) {
                V[r][c] = grid[r][c].reward;
            } else {
                // 对于普通状态，计算期望值
                std::vector<double> probs = policy_net.get_action_probs(r, c);
                for (int a = 0; a < ACTIONS; ++a) {
                    auto [next_r, next_c] = next_state(r, c, static_cast<Action>(a), grid);
                    V[r][c] += probs[a] * (grid[next_r][next_c].reward + GAMMA * V[next_r][next_c]);
                }
            }
        }
    }
}

#endif //REINFORCE_H

