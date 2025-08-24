//
// Created by cuihs on 2025/6/15.
//

#ifndef TRPO_H
#define TRPO_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../env/gridworld.h"
#include "../env/mdp_config.h"

// Trajectory structure for TRPO
struct TRPOTrajectory {
    std::vector<std::pair<int, int>> states;  // State sequence (r, c)
    std::vector<int> actions;                 // Action sequence
    std::vector<double> rewards;              // Reward sequence
    std::vector<double> action_probs;         // Action probabilities
    double total_return;                      // Total discounted return
};

// Policy network for TRPO
class TRPOPolicyNetwork {
private:
    std::vector<std::vector<std::vector<double>>> theta;  // Policy parameters [state_r][state_c][action]
    std::mt19937 rng;  // Random number generator
    
public:
    TRPOPolicyNetwork() : rng(std::random_device{}()) {
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
    
    // Compute policy gradient
    std::vector<std::vector<std::vector<double>>> compute_policy_gradient(
        const std::vector<TRPOTrajectory>& trajectories) {
        
        std::vector<std::vector<std::vector<double>>> gradients(
            ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
        
        for (const auto& traj : trajectories) {
            for (size_t t = 0; t < traj.states.size(); ++t) {
                int r = traj.states[t].first;
                int c = traj.states[t].second;
                int action = traj.actions[t];
                
                // Compute policy gradient
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
        
        return gradients;
    }
    
    // Compute Fisher Information Matrix (simplified diagonal approximation)
    std::vector<std::vector<std::vector<double>>> compute_fisher_info(
        const std::vector<TRPOTrajectory>& trajectories) {
        
        std::vector<std::vector<std::vector<double>>> fisher_info(
            ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
        
        for (const auto& traj : trajectories) {
            for (size_t t = 0; t < traj.states.size(); ++t) {
                int r = traj.states[t].first;
                int c = traj.states[t].second;
                int action = traj.actions[t];
                
                std::vector<double> probs = get_action_probs(r, c);
                // Diagonal Fisher information matrix
                fisher_info[r][c][action] += 1.0 / (probs[action] + 1e-8);
            }
        }
        
        return fisher_info;
    }
    
    // Update policy using TRPO
    void update_policy_trpo(const std::vector<TRPOTrajectory>& trajectories, 
                           double max_kl = 0.01, double damping = 0.1) {
        
        auto gradients = compute_policy_gradient(trajectories);
        auto fisher_info = compute_fisher_info(trajectories);
        
        // Compute natural gradient using Fisher information matrix
        std::vector<std::vector<std::vector<double>>> natural_gradients(
            ROWS, std::vector<std::vector<double>>(COLS, std::vector<double>(ACTIONS, 0.0)));
        
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                for (int a = 0; a < ACTIONS; ++a) {
                    if (fisher_info[r][c][a] > 1e-8) {
                        natural_gradients[r][c][a] = gradients[r][c][a] / (fisher_info[r][c][a] + damping);
                    }
                }
            }
        }
        
        // Compute step size using line search
        double step_size = compute_trpo_step_size(trajectories, natural_gradients, max_kl);
        
        // Update parameters
        for (int r = 0; r < ROWS; ++r) {
            for (int c = 0; c < COLS; ++c) {
                for (int a = 0; a < ACTIONS; ++a) {
                    theta[r][c][a] += step_size * natural_gradients[r][c][a];
                }
            }
        }
    }
    
    // Compute TRPO step size using line search
    double compute_trpo_step_size(const std::vector<TRPOTrajectory>& trajectories,
                                 const std::vector<std::vector<std::vector<double>>>& natural_gradients,
                                 double max_kl) {
        
        double step_size = 1.0;
        double alpha = 0.5;  // Backtracking factor
        
        // Line search to find appropriate step size
        for (int i = 0; i < 10; ++i) {
            // Compute KL divergence for current step size
            double kl_div = compute_kl_divergence(trajectories, natural_gradients, step_size);
            
            if (kl_div <= max_kl) {
                break;
            }
            
            step_size *= alpha;
        }
        
        return step_size;
    }
    
    // Compute KL divergence between old and new policy
    double compute_kl_divergence(const std::vector<TRPOTrajectory>& trajectories,
                                const std::vector<std::vector<std::vector<double>>>& natural_gradients,
                                double step_size) {
        
        double kl_div = 0.0;
        int count = 0;
        
        for (const auto& traj : trajectories) {
            for (size_t t = 0; t < traj.states.size(); ++t) {
                int r = traj.states[t].first;
                int c = traj.states[t].second;
                int action = traj.actions[t];
                
                // Old policy probability
                double old_prob = traj.action_probs[t];
                
                // New policy probability (approximated)
                std::vector<double> old_logits = theta[r][c];
                std::vector<double> new_logits = old_logits;
                for (int a = 0; a < ACTIONS; ++a) {
                    new_logits[a] += step_size * natural_gradients[r][c][a];
                }
                
                // Compute new probabilities
                double max_logit = *std::max_element(new_logits.begin(), new_logits.end());
                double sum_exp = 0.0;
                std::vector<double> new_probs(ACTIONS);
                for (int a = 0; a < ACTIONS; ++a) {
                    new_probs[a] = std::exp(new_logits[a] - max_logit);
                    sum_exp += new_probs[a];
                }
                for (int a = 0; a < ACTIONS; ++a) {
                    new_probs[a] /= sum_exp;
                }
                
                double new_prob = new_probs[action];
                
                // KL divergence
                if (old_prob > 1e-8 && new_prob > 1e-8) {
                    kl_div += old_prob * std::log(old_prob / new_prob);
                    count++;
                }
            }
        }
        
        return count > 0 ? kl_div / count : 0.0;
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

// Run episode and return trajectory
TRPOTrajectory run_episode_trpo(const Grid& grid, TRPOPolicyNetwork& policy_net, int max_steps = 1000) {
    TRPOTrajectory traj;
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
        
        // Record action probability
        double action_prob = policy_net.get_action_prob(r, c, action);
        traj.action_probs.push_back(action_prob);
        
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
    
    return traj;
}

// TRPO main function
void trpo(const Grid& grid, 
          std::vector<std::vector<double>>& V, 
          std::vector<std::vector<int>>& policy,
          int num_episodes = 1000,
          int episodes_per_update = 20,
          double max_kl = 0.01) {
    
    TRPOPolicyNetwork policy_net;
    std::vector<TRPOTrajectory> trajectories;
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        // Run episode
        TRPOTrajectory traj = run_episode_trpo(grid, policy_net);
        trajectories.push_back(traj);
        
        // Update policy every few episodes
        if ((episode + 1) % episodes_per_update == 0) {
            policy_net.update_policy_trpo(trajectories, max_kl);
            trajectories.clear();
        }
    }
    
    // Get final optimal policy
    policy = policy_net.get_optimal_policy();
    
    // Compute state value function
    V.assign(ROWS, std::vector<double>(COLS, 0.0));
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            if (grid[r][c].type == StateType::Terminal) {
                V[r][c] = grid[r][c].reward;
            } else if (grid[r][c].type == StateType::Forbidden) {
                V[r][c] = grid[r][c].reward;
            } else {
                // Compute expected value for normal states
                std::vector<double> probs = policy_net.get_action_probs(r, c);
                for (int a = 0; a < ACTIONS; ++a) {
                    auto [next_r, next_c] = next_state(r, c, static_cast<Action>(a), grid);
                    V[r][c] += probs[a] * (grid[next_r][next_c].reward + GAMMA * V[next_r][next_c]);
                }
            }
        }
    }
}

#endif //TRPO_H
