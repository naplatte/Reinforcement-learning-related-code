//
// Created by cuihs on 2025/6/15.
//

#ifndef PPO_H
#define PPO_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../env/gridworld.h"
#include "../env/mdp_config.h"

// Trajectory structure for PPO
struct PPOTrajectory {
    std::vector<std::pair<int, int>> states;  // State sequence (r, c)
    std::vector<int> actions;                 // Action sequence
    std::vector<double> rewards;              // Reward sequence
    std::vector<double> old_action_probs;     // Old action probabilities
    std::vector<double> advantages;           // Advantage estimates
    double total_return;                      // Total discounted return
};

// Policy network for PPO
class PPOPolicyNetwork {
private:
    std::vector<std::vector<std::vector<double>>> theta;  // Policy parameters [state_r][state_c][action]
    std::mt19937 rng;  // Random number generator
    
public:
    PPOPolicyNetwork() : rng(std::random_device{}()) {
        // Initialize policy parameters
        theta.assign(ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
    }
    
    // Get action probability distribution
    std::vector<double> get_action_probs(int r, int c) {
        std::vector<double> logits = theta[r][c];
        std::vector<double> probs(ACTIONS);
        
        // Compute softmax with numerical stability
        double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (int a = 0; a < ACTIONS; ++a) {
            probs[a] = std::exp(logits[a] - max_logit);
            sum_exp += probs[a];
        }
        
        // Normalize
        for (int a = 0; a < ACTIONS; ++a) {
            probs[a] /= sum_exp;
        }
        
        return probs;
    }
    
    // Sample action according to policy
    int sample_action(int r, int c) {
        std::vector<double> probs = get_action_probs(r, c);
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }
    
    // Get action probability
    double get_action_prob(int r, int c, int action) {
        std::vector<double> probs = get_action_probs(r, c);
        return probs[action];
    }
    
    // Compute PPO loss with clipping
    double compute_ppo_loss(const std::vector<PPOTrajectory>& trajectories, 
                           double epsilon = 0.2) {
        
        double total_loss = 0.0;
        int total_samples = 0;
        
        for (const auto& traj : trajectories) {
            for (size_t t = 0; t < traj.states.size(); ++t) {
                int r = traj.states[t].first;
                int c = traj.states[t].second;
                int action = traj.actions[t];
                
                double old_prob = traj.old_action_probs[t];
                double new_prob = get_action_prob(r, c, action);
                double advantage = traj.advantages[t];
                
                // Compute probability ratio
                double ratio = new_prob / (old_prob + 1e-8);
                
                // Clipped surrogate objective
                double clipped_ratio = std::clamp(ratio, 1.0 - epsilon, 1.0 + epsilon);
                double surrogate_loss = -std::min(ratio * advantage, clipped_ratio * advantage);
                
                total_loss += surrogate_loss;
                total_samples++;
            }
        }
        
        return total_samples > 0 ? total_loss / total_samples : 0.0;
    }
    
    // Update policy using PPO
    void update_policy_ppo(const std::vector<PPOTrajectory>& trajectories, 
                          double learning_rate = 0.001,
                          double epsilon = 0.2,
                          int num_epochs = 10) {
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Compute gradients for PPO loss
            std::vector<std::vector<std::vector<double>>> gradients(
                ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
            
            for (const auto& traj : trajectories) {
                for (size_t t = 0; t < traj.states.size(); ++t) {
                    int r = traj.states[t].first;
                    int c = traj.states[t].second;
                    int action = traj.actions[t];
                    
                    double old_prob = traj.old_action_probs[t];
                    double new_prob = get_action_prob(r, c, action);
                    double advantage = traj.advantages[t];
                    
                    // Compute probability ratio
                    double ratio = new_prob / (old_prob + 1e-8);
                    
                    // Clipped surrogate gradient
                    double clipped_ratio = std::clamp(ratio, 1.0 - epsilon, 1.0 + epsilon);
                    double gradient_scale = (ratio <= clipped_ratio) ? 1.0 : 0.0;
                    
                    // Compute policy gradient
                    std::vector<double> probs = get_action_probs(r, c);
                    for (int a = 0; a < ACTIONS; ++a) {
                        if (a == action) {
                            gradients[r][c][a] += gradient_scale * (1.0 - probs[a]) * advantage;
                        } else {
                            gradients[r][c][a] += gradient_scale * (-probs[a]) * advantage;
                        }
                    }
                }
            }
            
            // Update parameters
            for (int r = 0; r < ROWS; ++r) {
                for (int c = 0; c < COLS; ++c) {
                    for (int a = 0; a < ACTIONS; ++a) {
                        theta[r][c][a] += learning_rate * gradients[r][c][a];
                    }
                }
            }
        }
    }
    
    // Get optimal policy
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

// Value network for PPO (simplified)
class PPOValueNetwork {
private:
    std::vector<std::vector<double>> V;  // State values [r][c]
    
public:
    PPOValueNetwork() {
        V.assign(ROWS, std::vector<double>(COLS, 0.0));
    }
    
    // Get state value
    double get_value(int r, int c) {
        return V[r][c];
    }
    
    // Update value function using Monte Carlo returns
    void update_values(const std::vector<PPOTrajectory>& trajectories, double learning_rate = 0.001) {
        for (const auto& traj : trajectories) {
            double return_t = traj.total_return;
            for (size_t t = 0; t < traj.states.size(); ++t) {
                int r = traj.states[t].first;
                int c = traj.states[t].second;
                
                // Simple Monte Carlo update
                V[r][c] += learning_rate * (return_t - V[r][c]);
            }
        }
    }
    
    // Get all values
    std::vector<std::vector<double>> get_values() {
        return V;
    }
};

// Run episode and return trajectory
PPOTrajectory run_episode_ppo(const Grid& grid, PPOPolicyNetwork& policy_net, 
                             PPOValueNetwork& value_net, int max_steps = 1000) {
    PPOTrajectory traj;
    traj.total_return = 0.0;
    
    // Random starting state (avoid forbidden areas)
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
        // Record current state
        traj.states.emplace_back(r, c);
        
        // Sample action
        int action = policy_net.sample_action(r, c);
        traj.actions.push_back(action);
        
        // Record old action probability
        double action_prob = policy_net.get_action_prob(r, c, action);
        traj.old_action_probs.push_back(action_prob);
        
        // Execute action
        auto [next_r, next_c] = next_state(r, c, static_cast<Action>(action), grid);
        
        // Get reward
        double reward = grid[next_r][next_c].reward;
        traj.rewards.push_back(reward);
        
        // Accumulate discounted return
        traj.total_return += gamma_power * reward;
        gamma_power *= GAMMA;
        
        // Check if reached terminal state
        if (grid[next_r][next_c].type == StateType::Terminal) {
            break;
        }
        
        // Check if entered forbidden area
        if (grid[next_r][next_c].type == StateType::Forbidden) {
            break;
        }
        
        r = next_r;
        c = next_c;
    }
    
    // Compute advantages (Monte Carlo advantage)
    traj.advantages.resize(traj.states.size());
    double advantage = traj.total_return;
    for (size_t t = 0; t < traj.states.size(); ++t) {
        int r = traj.states[t].first;
        int c = traj.states[t].second;
        double value = value_net.get_value(r, c);
        traj.advantages[t] = advantage - value;
        advantage -= traj.rewards[t];
    }
    
    return traj;
}

// PPO main function
void ppo(const Grid& grid, 
         std::vector<std::vector<double>>& V, 
         std::vector<std::vector<int>>& policy,
         int num_episodes = 1000,
         int episodes_per_update = 20,
         double learning_rate = 0.001,
         double epsilon = 0.2) {
    
    PPOPolicyNetwork policy_net;
    PPOValueNetwork value_net;
    std::vector<PPOTrajectory> trajectories;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        // Run episode
        PPOTrajectory traj = run_episode_ppo(grid, policy_net, value_net);
        trajectories.push_back(traj);
        
        // Update policy and value function every few episodes
        if ((episode + 1) % episodes_per_update == 0) {
            // Update value function
            value_net.update_values(trajectories, learning_rate);
            
            // Update policy using PPO
            policy_net.update_policy_ppo(trajectories, learning_rate, epsilon);
            
            trajectories.clear();
        }
    }
    
    // Get final optimal policy
    policy = policy_net.get_optimal_policy();
    
    // Get final state values
    V = value_net.get_values();
    
    // Update terminal and forbidden state values
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (grid[r][c].type == StateType::Terminal) {
                V[r][c] = grid[r][c].reward;
            } else if (grid[r][c].type == StateType::Forbidden) {
                V[r][c] = grid[r][c].reward;
            }
        }
    }
}

#endif //PPO_H
