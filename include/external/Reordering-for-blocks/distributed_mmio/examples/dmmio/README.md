# Distributed Matrix Market Read

## Build 

```bash
cmake -B build
cmake --build build
```

## Run

```bash
# Local experiment
mpirun -np 8 --map-by :OVERSUBSCRIBE ./build/main -r 2 -c 2 -f ../matrices/16x16_dense.bmtx -p 1 -t 0
```