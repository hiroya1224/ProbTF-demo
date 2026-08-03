cmake_minimum_required(VERSION 3.0.2)
project(grape_param_estim)

find_package(catkin REQUIRED)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/grape_assimilate_flights.py
    scripts/grape_evaluate_pid_proposals.py
    scripts/grape_generate_synthetic_flight.py
    scripts/grape_inspect_flights.py
    scripts/grape_run_strong_constraint_experiment.py
    scripts/grape_run_weak_constraint_experiment.py
    scripts/grape_validate_assimilation.py
    scripts/grape_validate_held_out_flight.py
    gui/run_gui.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  FILES
    README.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  DIRECTORY
    samples/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/samples
  USE_SOURCE_PERMISSIONS
)

install(
  DIRECTORY
    gui/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/gui
  USE_SOURCE_PERMISSIONS
  PATTERN ".venv" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests/grape_param_estim
  )
endif()
