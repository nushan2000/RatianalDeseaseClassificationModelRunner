//openmp
g++ -fopenmp -I./onnxruntime-linux-x64-1.17.0/include     test1.cpp     -L./onnxruntime-linux-x64-1.17.0/lib     ./onnxruntime-linux-x64-1.17.0/lib/libonnxruntime.so     -pthread     `pkg-config --cflags --libs opencv4`     -o test

export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.17.0/lib:$LD_LIBRARY_PATH

./test

//mpi
mpic++ -std=c++17 -I./onnxruntime-linux-x64-1.17.0/include -L./onnxruntime-linux-x64-1.17.0/lib -o infer testmpi.cpp `pkg-config --cflags --libs opencv4` -lonnxruntime
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.17.0/lib:$LD_LIBRARY_PATH
mpirun -np 4 ./infer ./images


//for single image code. test5.cpp
g++ test5.cpp -o test5   -I./onnxruntime-linux-x64-1.17.0/include   -L./onnxruntime-linux-x64-1.17.0/lib   -lonnxruntime   `pkg-config --cflags --libs opencv4`   -std=c++17

export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.17.0/lib:$LD_LIBRARY_PATH


//serial code
g++ -I./onnxruntime-linux-x64-1.17.0/include serialTest.cpp \
    -L./onnxruntime-linux-x64-1.17.0/lib \
    ./onnxruntime-linux-x64-1.17.0/lib/libonnxruntime.so \
    -pthread `pkg-config --cflags --libs opencv4` \
    -o serialTest
    
    export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.17.0/lib:$LD_LIBRARY_PATH
    
    ./serialTest



//hybrid
mpicxx -fopenmp -I./onnxruntime-linux-x64-1.17.0/include \
  hybrid.cpp \
  -L./onnxruntime-linux-x64-1.17.0/lib \
  ./onnxruntime-linux-x64-1.17.0/lib/libonnxruntime.so \
  -pthread \
  `pkg-config --cflags --libs opencv4` \
  -o hybrid
export LD_LIBRARY_PATH=./onnxruntime-linux-x64-1.17.0/lib:$LD_LIBRARY_PATH
# Example: 4 MPI processes, each using 2 OpenMP threads
export OMP_NUM_THREADS=2
mpirun -np 4 ./hybrid ./images







