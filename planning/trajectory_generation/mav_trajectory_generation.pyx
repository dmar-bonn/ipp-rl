import numpy as np
from planning.trajectory_generation.mav_trajectory_generation cimport TrajectoryPlanner
from libcpp.vector cimport vector

cdef class MavTrajectoryGenerator:
    cdef TrajectoryPlanner* c_planner

    def __cinit__(self, double max_v, double max_a):
        self.c_planner = new TrajectoryPlanner(max_v, max_a)

    def __dealloc__(self):
        del self.c_planner

    def plan_uav_trajectory(self, vertices_constraints: np.array, sampling_time: float = 0.01) -> np.array:
        cdef vector[vector[double]] c_vertices_constraints
        cdef vector[vector[double]] c_trajectory_samples
        cdef vector[double] c_vertex_constraint
        cdef vector[double] c_trajectory_sample
        cdef double c_sampling_time
        cdef int i

        c_vertex_constraint.push_back(vertices_constraints[0][0])
        c_vertex_constraint.push_back(vertices_constraints[0][1])
        c_vertex_constraint.push_back(vertices_constraints[0][2])
        c_vertices_constraints.push_back(c_vertex_constraint)

        for i in range(1, vertices_constraints.shape[0]):
            c_vertex_constraint[0] = vertices_constraints[i][0]
            c_vertex_constraint[1] = vertices_constraints[i][1]
            c_vertex_constraint[2] = vertices_constraints[i][2]
            c_vertices_constraints.push_back(c_vertex_constraint)

        c_sampling_time = sampling_time
        c_trajectory_samples = self.c_planner.planTrajectory(c_vertices_constraints, c_sampling_time)

        py_trajectory_samples = []
        for i in range(c_trajectory_samples.size()):
            c_trajectory_sample = c_trajectory_samples[i]
            np_trajectory_sample = np.array([c_trajectory_sample[0], c_trajectory_sample[1], c_trajectory_sample[2]])
            py_trajectory_samples.append(np_trajectory_sample)

        return np.array(py_trajectory_samples)
