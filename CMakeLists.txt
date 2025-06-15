cmake_minimum_required(VERSION 3.31)
project(Reinforcement_learning_related_code)

set(CMAKE_CXX_STANDARD 20)

add_executable(Reinforcement_learning_related_code main.cpp
        "Value Iteration and Policy Iteration/Value_Iteration.h"
        env/mdp_config.h
        env/gridworld.h)
