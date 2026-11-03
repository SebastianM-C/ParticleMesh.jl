# The environment of Smilei's GPU install guide (doc/Sphinx/Use/install_linux_GPU.rst at v5.1,
# "Create a file nvidia_env.sh"), for the NVIDIA HPC SDK installed from NVIDIA's apt repository
# under /opt/nvidia/hpc_sdk. Sourced by the image build with four variables set:
#   PM_GPU_CC    the -gpu=cc target of nvc++ (cc89, or cc86 if this nvc++ refuses cc89)
#   PM_CUDA      the bundled CUDA release, named in every -gpu= (-gpu=cuda12.3): without a
#                driver nvc++ otherwise picks the driver's CUDA or its fallback, 11.8
#   PM_SM        the -arch=sm_ target of nvcc (89: the RTX 4080 SUPER)
#   PM_FASTMATH  ",fastmath" (the guide's flags) or "" (the same flags without it)
# Departures from the guide, each because the guide's value cannot work in an image build:
#   -gpu=cuda12.3 names the toolkit, which the guide leaves to the driver the build machine has.
#   -gpu=ccnative and -arch=sm_86 name the build machine's GPU; the build has none, so the target
#   is named. nvcc's --compiler-bindir $NVDIR points at a directory without a host compiler; nvcc
#   uses the system g++ (11, within the guide's GCC <= 12). The MPI include path is asked of the
#   SDK's own wrapper instead of being spelled with its Open MPI version. -cuda is added to the
#   link line so that the CUDA runtime of the nvcc-compiled .cu kernels is linked explicitly
#   (Smilei's ruche_gpu2 machine file does the same).
NV=/opt/nvidia/hpc_sdk/Linux_x86_64/${NVHPC_RELEASE:?}
export PATH=$NV/compilers/bin:$NV/comm_libs/mpi/bin:$PATH

export HDF5_ROOT_DIR=/opt/hdf5
export LD_LIBRARY_PATH=$HDF5_ROOT_DIR/lib

MPI_INC=$(mpicxx --showme:incdirs | tr ' ' '\n' | sed '/^$/d; s/^/-I/' | tr '\n' ' ')

export LDFLAGS="-acc=gpu -gpu=${PM_GPU_CC:?},cuda${PM_CUDA:?} -cudalib=curand -cuda "
export CXXFLAGS="-acc=gpu -gpu=${PM_GPU_CC},cuda${PM_CUDA}${PM_FASTMATH} -std=c++14 -lcurand -Minfo=accel -w -D__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1 -I$NV/math_libs/include/"
export GPU_COMPILER_FLAGS="-O3 --std c++14 -arch=sm_${PM_SM:?} --expt-relaxed-constexpr $MPI_INC -I$HDF5_ROOT_DIR/include/"

export SMILEICXX_DEPS=g++

export SLURM_LOCALID=0
