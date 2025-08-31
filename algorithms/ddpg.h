//
// Created by cuihs on 2025/6/15.
//

#ifndef DDPG_H
#define DDPG_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include "../env/gridworld.h"
#include "../env/mdp_config.h"

// Experience replay buffer for DDPG
struct DDPGExperience {
    std::pair<int, int> state;      // Current state (r, c)
    int action;                     // Action taken
    double reward;                  // Reward received
    std::pair<int, int> next_state; // Next state (r, c)
    bool done;                      // Whether episode ended
    
    DDPGExperience(std::pair<int, int> s, int a, double r, std::pair<int, int> ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
};

// Experience Replay Buffer
class ReplayBuffer {
private:
    std::deque<DDPGExperience> buffer;
    size_t max_size;
    std::mt19937 rng;
    
public:
    ReplayBuffer(size_t capacity = 10000) : max_size(capacity), rng(std::random_device{}()) {}
    
    void push(const DDPGExperience& exp) {
        if (buffer.size() >= max_size) {
            buffer.pop_front();
        }
        buffer.push_back(exp);
    }
    
    std::vector<DDPGExperience> sample(size_t batch_size) {
        std::vector<DDPGExperience> batch;
        if (buffer.size() < batch_size) {
            batch_size = buffer.size();
        }
        
        std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = dist(rng);
            batch.push_back(buffer[idx]);
        }
        return batch;
    }
    
    size_t size() const { return buffer.size(); }
};

// Actor Network (Policy Network)
class DDPGActor {
private:
    std::vector<std::vector<std::vector<double>>> theta;  // Actor parameters [state_r][state_c][action]
    std::vector<std::vector<std::vector<double>>> target_theta;  // Target network parameters
    std::mt19937 rng;
    
public:
    DDPGActor() : rng(std::random_device{}()) {
        // Initialize actor parameters
        theta.assign(ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
        target_theta = theta;
    }
    
    // Get action probabilities (softmax)
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
    
    // Get deterministic action (argmax)
    int get_action(int r, int c) {
        std::vector<double> probs = get_action_probs(r, c);
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }
    
    // Get action with exploration noise
    int get_action_with_noise(int r, int c, double epsilon = 0.1) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) {
            // Random action
            std::uniform_int_distribution<int> action_dist(0, ACTIONS - 1);
            return action_dist(rng);
        } else {
            // Deterministic action
            return get_action(r, c);
        }
    }
    
    // Get action probability
    double get_action_prob(int r, int c, int action) {
        std::vector<double> probs = get_action_probs(r, c);
        return probs[action];
    }
    
    // Update actor parameters
    void update_actor(const std::vector<DDPGExperience>& batch, 
                     const std::vector<std::vector<std::vector<double>>>& q_gradients,
                     double lr = 0.001) {
        
        for (const auto& exp : batch) {
            int r = exp.state.first;
            int c = exp.state.second;
            int action = exp.action;
            
            // Update actor parameters using Q-function gradients
            std::vector<double> probs = get_action_probs(r, c);
            for (int a = 0; a < ACTIONS; ++a) {
                if (a == action) {
                    theta[r][c][a] += lr * q_gradients[r][c][a] * (1.0 - probs[a]);
                } else {
                    theta[r][c][a] += lr * q_gradients[r][c][a] * (-probs[a]);
                }
            }
        }
    }
    
    // Update target network
    void update_target(double tau = 0.001) {
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                for (int a = 0; a < ACTIONS; ++a) {
                    target_theta[r][c][a] = tau * theta[r][c][a] + (1.0 - tau) * target_theta[r][c][a];
                }
            }
        }
    }
    
    // Get optimal policy
    std::vector<std::vector<int>> get_optimal_policy() {
        std::vector<std::vector<int>> policy(ROWS, std::vector<int>(COLS));
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                policy[r][c] = get_action(r, c);
            }
        }
        return policy;
    }
};

// Critic Network (Q-function)
class DDPGCritic {
private:
    std::vector<std::vector<std::vector<double>>> Q;  // Q-function [state_r][state_c][action]
    std::vector<std::vector<std::vector<double>>> target_Q;  // Target Q-function
    std::mt19937 rng;
    
public:
    DDPGCritic() : rng(std::random_device{}()) {
        // Initialize Q-function
        Q.assign(ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
        target_Q = Q;
    }
    
    // Get Q-value
    double get_q_value(int r, int c, int action) {
        return Q[r][c][action];
    }
    
    // Get max Q-value for a state
    double get_max_q_value(int r, int c) {
        return *std::max_element(Q[r][c].begin(), Q[r][c].end());
    }
    
    // Get target max Q-value for a state
    double get_target_max_q_value(int r, int c) {
        return *std::max_element(target_Q[r][c].begin(), target_Q[r][c].end());
    }
    
    // Update Q-function
    void update_critic(const std::vector<DDPGExperience>& batch, double lr = 0.001) {
        for (const auto& exp : batch) {
            int r = exp.state.first;
            int c = exp.state.second;
            int action = exp.action;
            int next_r = exp.next_state.first;
            int next_c = exp.next_state.second;
            
            // Compute target Q-value
            double target_q;
            if (exp.done) {
                target_q = exp.reward;
            } else {
                target_q = exp.reward + GAMMA * get_target_max_q_value(next_r, next_c);
            }
            
            // Update Q-value
            Q[r][c][action] += lr * (target_q - Q[r][c][action]);
        }
    }
    
    // Update target network
    void update_target(double tau = 0.001) {
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                for (int a = 0; a < ACTIONS; ++a) {
                    target_Q[r][c][a] = tau * Q[r][c][a] + (1.0 - tau) * target_Q[r][c][a];
                }
            }
        }
    }
    
    // Compute gradients for actor update
    std::vector<std::vector<std::vector<double>>> compute_q_gradients(int r, int c) {
        std::vector<std::vector<std::vector<double>>> gradients(
            ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
        
        // For discrete actions, we use the Q-values directly as gradients
        for (int a = 0; a < ACTIONS; ++a) {
            gradients[r][c][a] = Q[r][c][a];
        }
        
        return gradients;
    }
};

// Run episode and collect experiences
std::vector<DDPGExperience> run_episode_ddpg(const Grid& grid, DDPGActor& actor, 
                                            ReplayBuffer& replay_buffer, int max_steps = 1000) {
    std::vector<DDPGExperience> episode_experiences;
    
    // Random starting state (avoid forbidden areas)
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_r(0, ROWS - 1);
    std::uniform_int_distribution<int> dist_c(0, COLS - 1);
    
    int r, c;
    do {
        r = dist_r(rng);
        c = dist_c(rng);
    } while (grid[r][c].type == StateType::Forbidden);
    
    for (int step = 0; step < max_steps; ++step) {
        // Get action with exploration
        int action = actor.get_action_with_noise(r, c, 0.1);
        
        // Execute action
        auto [next_r, next_c] = next_state(r, c, static_cast<Action>(action), grid);
        
        // Get reward
        double reward = grid[next_r][next_c].reward;
        
        // Check if episode ended
        bool done = (grid[next_r][next_c].type == StateType::Terminal || 
                    grid[next_r][next_c].type == StateType::Forbidden);
        
        // Store experience
        DDPGExperience exp({r, c}, action, reward, {next_r, next_c}, done);
        episode_experiences.push_back(exp);
        replay_buffer.push(exp);
        
        if (done) {
            break;
        }
        
        r = next_r;
        c = next_c;
    }
    
    return episode_experiences;
}

// DDPG main function
void ddpg(const Grid& grid, 
          std::vector<std::vector<double>>& V, 
          std::vector<std::vector<int>>& policy,
          int num_episodes = 1000,
          int batch_size = 32,
          double actor_lr = 0.001,
          double critic_lr = 0.001,
          double tau = 0.001) {
    
    DDPGActor actor;
    DDPGCritic critic;
    ReplayBuffer replay_buffer(10000);
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        // Run episode and collect experiences
        auto episode_experiences = run_episode_ddpg(grid, actor, replay_buffer);
        
        // Update networks if enough experiences
        if (replay_buffer.size() >= batch_size) {
            // Sample batch from replay buffer
            auto batch = replay_buffer.sample(batch_size);
            
            // Update critic
            critic.update_critic(batch, critic_lr);
            
            // Update actor
            for (const auto& exp : batch) {
                auto q_gradients = critic.compute_q_gradients(exp.state.first, exp.state.second);
                std::vector<DDPGExperience> single_exp = {exp};
                actor.update_actor(single_exp, q_gradients, actor_lr);
            }
            
            // Update target networks
            actor.update_target(tau);
            critic.update_target(tau);
        }
    }
    
    // Get final optimal policy
    policy = actor.get_optimal_policy();
    
    // Compute state value function
    V.assign(ROWS, std::vector<double>(COLS, 0.0));
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (grid[r][c].type == StateType::Terminal) {
                V[r][c] = grid[r][c].reward;
            } else if (grid[r][c].type == StateType::Forbidden) {
                V[r][c] = grid[r][c].reward;
            } else {
                // Use max Q-value as state value
                V[r][c] = critic.get_max_q_value(r, c);
            }
        }
    }
}

#endif //DDPG_H
