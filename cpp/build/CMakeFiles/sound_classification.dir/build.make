# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/cx/soundclassification/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/cx/soundclassification/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/sound_classification.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sound_classification.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sound_classification.dir/flags.make

CMakeFiles/sound_classification.dir/bind.cpp.o: CMakeFiles/sound_classification.dir/flags.make
CMakeFiles/sound_classification.dir/bind.cpp.o: ../bind.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/cx/soundclassification/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sound_classification.dir/bind.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sound_classification.dir/bind.cpp.o -c /data/cx/soundclassification/cpp/bind.cpp

CMakeFiles/sound_classification.dir/bind.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sound_classification.dir/bind.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/cx/soundclassification/cpp/bind.cpp > CMakeFiles/sound_classification.dir/bind.cpp.i

CMakeFiles/sound_classification.dir/bind.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sound_classification.dir/bind.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/cx/soundclassification/cpp/bind.cpp -o CMakeFiles/sound_classification.dir/bind.cpp.s

CMakeFiles/sound_classification.dir/melspec.cpp.o: CMakeFiles/sound_classification.dir/flags.make
CMakeFiles/sound_classification.dir/melspec.cpp.o: ../melspec.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/cx/soundclassification/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sound_classification.dir/melspec.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sound_classification.dir/melspec.cpp.o -c /data/cx/soundclassification/cpp/melspec.cpp

CMakeFiles/sound_classification.dir/melspec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sound_classification.dir/melspec.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/cx/soundclassification/cpp/melspec.cpp > CMakeFiles/sound_classification.dir/melspec.cpp.i

CMakeFiles/sound_classification.dir/melspec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sound_classification.dir/melspec.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/cx/soundclassification/cpp/melspec.cpp -o CMakeFiles/sound_classification.dir/melspec.cpp.s

CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.o: CMakeFiles/sound_classification.dir/flags.make
CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.o: ../sound_classification_v2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/cx/soundclassification/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.o -c /data/cx/soundclassification/cpp/sound_classification_v2.cpp

CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/cx/soundclassification/cpp/sound_classification_v2.cpp > CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.i

CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/cx/soundclassification/cpp/sound_classification_v2.cpp -o CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.s

# Object files for target sound_classification
sound_classification_OBJECTS = \
"CMakeFiles/sound_classification.dir/bind.cpp.o" \
"CMakeFiles/sound_classification.dir/melspec.cpp.o" \
"CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.o"

# External object files for target sound_classification
sound_classification_EXTERNAL_OBJECTS =

../sound_classification.so: CMakeFiles/sound_classification.dir/bind.cpp.o
../sound_classification.so: CMakeFiles/sound_classification.dir/melspec.cpp.o
../sound_classification.so: CMakeFiles/sound_classification.dir/sound_classification_v2.cpp.o
../sound_classification.so: CMakeFiles/sound_classification.dir/build.make
../sound_classification.so: /opt/sophon/libsophon-current/lib/libbmlib.so
../sound_classification.so: /opt/sophon/libsophon-current/lib/libbmrt.so
../sound_classification.so: /opt/sophon/libsophon-current/lib/libbmcv.so
../sound_classification.so: CMakeFiles/sound_classification.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/cx/soundclassification/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared module ../sound_classification.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sound_classification.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sound_classification.dir/build: ../sound_classification.so

.PHONY : CMakeFiles/sound_classification.dir/build

CMakeFiles/sound_classification.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sound_classification.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sound_classification.dir/clean

CMakeFiles/sound_classification.dir/depend:
	cd /data/cx/soundclassification/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/cx/soundclassification/cpp /data/cx/soundclassification/cpp /data/cx/soundclassification/cpp/build /data/cx/soundclassification/cpp/build /data/cx/soundclassification/cpp/build/CMakeFiles/sound_classification.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sound_classification.dir/depend
