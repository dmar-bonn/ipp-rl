from libcpp.vector cimport vector


cdef extern from "/mapping_ipp_framework/src/mav_trajectory_generation/mav_trajectory_generation/include/mav_trajectory_generation/trajectory_planner.h" namespace "mav_trajectory_generation":
    cdef cppclass TrajectoryPlanner:
        TrajectoryPlanner(double, double) except +
        vector[vector[double]] planTrajectory(vector[vector[double]] vertices_constraints, double sampling_interval)
