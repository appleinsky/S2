# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/ma-user/work/cmake-3.28.3-linux-aarch64/bin/cmake

# The command to remove a file.
RM = /home/ma-user/work/cmake-3.28.3-linux-aarch64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ma-user/work/suanzi/BallQuery

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ma-user/work/suanzi/BallQuery/build_out

# Utility rule file for ascendc_bin_ascend910b_ball_query_copy.

# Include any custom commands dependencies for this target.
include op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/compiler_depend.make

# Include the progress variables for this target.
include op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/progress.make

op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy:
	cd /home/ma-user/work/suanzi/BallQuery/build_out/op_kernel && cp /home/ma-user/work/suanzi/BallQuery/build_out/op_kernel/tbe/dynamic/ball_query.py /home/ma-user/work/suanzi/BallQuery/build_out/op_kernel/binary/ascend910b/src/BallQuery.py

ascendc_bin_ascend910b_ball_query_copy: op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy
ascendc_bin_ascend910b_ball_query_copy: op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/build.make
.PHONY : ascendc_bin_ascend910b_ball_query_copy

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/build: ascendc_bin_ascend910b_ball_query_copy
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/build

op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/clean:
	cd /home/ma-user/work/suanzi/BallQuery/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/clean

op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/depend:
	cd /home/ma-user/work/suanzi/BallQuery/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/work/suanzi/BallQuery /home/ma-user/work/suanzi/BallQuery/op_kernel /home/ma-user/work/suanzi/BallQuery/build_out /home/ma-user/work/suanzi/BallQuery/build_out/op_kernel /home/ma-user/work/suanzi/BallQuery/build_out/op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend910b_ball_query_copy.dir/depend

