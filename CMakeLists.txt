cmake_minimum_required(VERSION 3.31)
project(Reinforcement_learning_related_code)

set(CMAKE_CXX_STANDARD 20)

add_executable(Reinforcement_learning_related_code main.cpp
        env/mdp_config.h
        env/gridworld.h
        env/gridworld.cpp
        algorithms/value_iteration.h
        algorithms/policy_iteration.h)
