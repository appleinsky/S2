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
CMAKE_SOURCE_DIR = /home/ma-user/work/suanzi/DepthToSpace

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ma-user/work/suanzi/DepthToSpace/build_out

# Utility rule file for ops_info_gen_ascend910b.

# Include any custom commands dependencies for this target.
include op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/compiler_depend.make

# Include the progress variables for this target.
include op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/progress.make

op_kernel/CMakeFiles/ops_info_gen_ascend910b: op_kernel/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json

op_kernel/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/ma-user/work/suanzi/DepthToSpace/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json"
	cd /home/ma-user/work/suanzi/DepthToSpace/build_out/op_kernel && mkdir -p /home/ma-user/work/suanzi/DepthToSpace/build_out/op_kernel/tbe/op_info_cfg/ai_core/ascend910b
	cd /home/ma-user/work/suanzi/DepthToSpace/build_out/op_kernel && python3 /home/ma-user/work/suanzi/DepthToSpace/cmake/util/parse_ini_to_json.py /home/ma-user/work/suanzi/DepthToSpace/build_out/autogen/aic-ascend910b-ops-info.ini /home/ma-user/work/suanzi/DepthToSpace/build_out/op_kernel/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json

ops_info_gen_ascend910b: op_kernel/CMakeFiles/ops_info_gen_ascend910b
ops_info_gen_ascend910b: op_kernel/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json
ops_info_gen_ascend910b: op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/build.make
.PHONY : ops_info_gen_ascend910b

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/build: ops_info_gen_ascend910b
.PHONY : op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/build

op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/clean:
	cd /home/ma-user/work/suanzi/DepthToSpace/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/ops_info_gen_ascend910b.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/clean

op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/depend:
	cd /home/ma-user/work/suanzi/DepthToSpace/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/work/suanzi/DepthToSpace /home/ma-user/work/suanzi/DepthToSpace/op_kernel /home/ma-user/work/suanzi/DepthToSpace/build_out /home/ma-user/work/suanzi/DepthToSpace/build_out/op_kernel /home/ma-user/work/suanzi/DepthToSpace/build_out/op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : op_kernel/CMakeFiles/ops_info_gen_ascend910b.dir/depend

