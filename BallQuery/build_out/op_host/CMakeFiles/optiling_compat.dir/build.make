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

# Utility rule file for optiling_compat.

# Include any custom commands dependencies for this target.
include op_host/CMakeFiles/optiling_compat.dir/compiler_depend.make

# Include the progress variables for this target.
include op_host/CMakeFiles/optiling_compat.dir/progress.make

op_host/CMakeFiles/optiling_compat:
	cd /home/ma-user/work/suanzi/BallQuery/build_out/op_host && ln -sf lib/linux/aarch64/libcust_opmaster_rt2.0.so /home/ma-user/work/suanzi/BallQuery/build_out/op_host/liboptiling.so

optiling_compat: op_host/CMakeFiles/optiling_compat
optiling_compat: op_host/CMakeFiles/optiling_compat.dir/build.make
.PHONY : optiling_compat

# Rule to build all files generated by this target.
op_host/CMakeFiles/optiling_compat.dir/build: optiling_compat
.PHONY : op_host/CMakeFiles/optiling_compat.dir/build

op_host/CMakeFiles/optiling_compat.dir/clean:
	cd /home/ma-user/work/suanzi/BallQuery/build_out/op_host && $(CMAKE_COMMAND) -P CMakeFiles/optiling_compat.dir/cmake_clean.cmake
.PHONY : op_host/CMakeFiles/optiling_compat.dir/clean

op_host/CMakeFiles/optiling_compat.dir/depend:
	cd /home/ma-user/work/suanzi/BallQuery/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/work/suanzi/BallQuery /home/ma-user/work/suanzi/BallQuery/op_host /home/ma-user/work/suanzi/BallQuery/build_out /home/ma-user/work/suanzi/BallQuery/build_out/op_host /home/ma-user/work/suanzi/BallQuery/build_out/op_host/CMakeFiles/optiling_compat.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : op_host/CMakeFiles/optiling_compat.dir/depend

