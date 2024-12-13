cmake_minimum_required(VERSION 3.12.0)
project(tao_akantu_binding)

find_package(Akantu REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)

add_akantu_simulation(main main.cc tao_solver.cc
    FILES_TO_COPY input/material.dat input/circle_R0.01_P2.msh
)