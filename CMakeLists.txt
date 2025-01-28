cmake_minimum_required(VERSION 3.12.0)
project(tao_akantu_binding)

find_package(Akantu REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)

add_akantu_simulation(contact contact_gen.cc
    FILES_TO_COPY input/material.dat input/square_L0.01_P1_dy0_lc0.001.msh
)