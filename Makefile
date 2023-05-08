########################################################################
#                          -*- Makefile -*-                            #
########################################################################
# Force rebuild on these rules
.PHONY: all libs clean clean-libs
.DEFAULT_GOAL := main

COMPILER = g++

########################################################################
## Flags
FLAGS   = -g -std=c++17
#FLAGS   = -g -std=c++17
## find shared libraries during runtime: set rpath:
LDFLAGS = -rpath @executable_path/libs
PREPRO  =
##verbose level 1
#DEBUG   = -D DEBUGV1
##verbose level 2
#DEBUG  += -D DEBUGV2
##verbose level 3
#DEBUG  += -D DEBUGV3
OPT     = -O3
WARN    = -Wall -Wno-missing-braces

### generate directory obj, if not yet existing
$(shell mkdir -p build)

########################################################################
## Paths, modify if necessary
WORKINGDIR = $(shell pwd)
PARENTDIR  = $(WORKINGDIR)/..
SYSTEMINC  = /opt/local/include
LIBS       = $(WORKINGDIR)/libs
EIGENINT   = $(LIBS)/eigen-stl-interface/
EIGEN      = $(LIBS)/eigen
GTEST      = $(LIBS)/googletest
LIBGTEST   = $(LIBS)/googletest/build/lib

########################################################################
### DO NOT MODIFY BELOW THIS LINE ######################################
########################################################################

########################################################################
## search for the files and set paths
vpath %.cpp $(WORKINGDIR) $(WORKINGDIR)/unittests
vpath %.m $(WORKINGDIR)
vpath %.a $(WORKINGDIR)/build
vpath %.o $(WORKINGDIR)/build

########################################################################
## Includes
CXX  = $(COMPILER) $(FLAGS) $(OPT) $(WARN) $(DEBUG) $(PREPRO) -I$(SYSTEMINC) -I$(WORKINGDIR) -I$(LIBS) -I$(EIGENINT) -I$(EIGEN) -I$(GTEST)/googletest/include
INCLUDE = $(wildcard *.h $(UINCLUDE)/*.h)

########################################################################
## libraries
### libconfig
#LDFLAGS += -lconfig++

# Frameworks
# -framework SDL_gfx 
#FRM = -framework Cocoa

### Unittests
LDFLAGS_U = $(LDFLAGS)
LDFLAGS_U += -L$(LIBGTEST) -lgtest

########################################################################
## Build rules
%.a: %.cpp $(INCLUDE)
	$(CXX) -c -o build/$@ $<

%.a: %.m $(INCLUDE)
	$(CXX) -c -o build/$@ $<

########################################################################
## BUILD Files
BUILD = main.a

## BUILD files for unittests
BUILD_U = unittests.a gtest.a


########################################################################
## Rules
## type make -j4 [rule] to speed up the compilation
all: libs main gtest

main: $(BUILD)
	$(CXX) $(patsubst %,build/%,$(BUILD)) $(LDFLAGS) $(FRM) -o $@

gtest: $(BUILD_U)
	$(CXX) $(patsubst %,build/%,$(BUILD_U)) $(LDFLAGS_U) -o $@

libs:
	cd $(GTEST) && mkdir -p $(GTEST)/build && cd $(GTEST)/build && \
	cmake -DBUILD_SHARED_LIBS=ON .. && make
	cp $(LIBGTEST)/* $(LIBS)

clean-all: clean clean-libs

clean:
	rm -f build/*.a main gtest

clean-libs:
	cd $(GTEST) && rm -rf build 
	rm -f $(LIBS)/*.dylib $(LIBS)/*.so

do:
	make && ./main

########################################################################
#                       -*- End of Makefile -*-                        #
########################################################################