from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension(
        "planning.trajectory_generation.mav_trajectory_generation",
        sources=[
            "planning/trajectory_generation/mav_trajectory_generation.pyx",
            "src/mav_trajectory_generation/mav_trajectory_generation/src/trajectory_planner.cpp",
        ],
        include_dirs=[
            "/mapping_ipp_framework/src/mav_trajectory_generation/mav_trajectory_generation/include",
            "/mapping_ipp_framework/src/mav_trajectory_generation/mav_trajectory_generation/include",
            "/mapping_ipp_framework/src/mav_comm/mav_msgs/include",
            "/mapping_ipp_framework/devel/include/",
            "/opt/ros/noetic/include",
            "/usr/include/eigen3",
        ],
        language="c++",
        library_dirs=["devel/lib/"],
        runtime_library_dirs=["devel/lib/"],
        extra_objects=[
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/motion_defines.cpp.o",
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/polynomial.cpp.o",
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/segment.cpp.o",
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/timing.cpp.o",
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/trajectory.cpp.o",
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/trajectory_sampling.cpp.o",
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/vertex.cpp.o",
            "build/mav_trajectory_generation/CMakeFiles/mav_trajectory_generation.dir/src/rpoly/rpoly_ak1.cpp.o",
            "/mapping_ipp_framework/devel/lib/libnlopt_wrap.so",
            "/mapping_ipp_framework/devel/lib/libglog.so",
        ],
    )
]

setup(
    name="mapping_ipp_framework",
    version="0.1.0",
    description="Mapping and IPP framework",
    ext_modules=cythonize(extensions, language_level="3"),
)
