cmake_minimum_required(VERSION 3.0.2)
project(deflecomp_core)

find_package(catkin REQUIRED COMPONENTS probtf_core)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS probtf_core)
