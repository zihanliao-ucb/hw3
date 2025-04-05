Need to compile with:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=CC ..
cmake --build .
```


salloc -N 1 -A mp309 -t 03:00 -q debug --qos=interactive -C cpu

srun -N 1 -n 32 ./kmer_hash_51 /pscratch/sd/z/zihan02/hw3_datasets/human-chr14-synthetic.txt test
cat test*.dat | sort > my_solution.txt
diff my_solution.txt /pscratch/sd/z/zihan02/hw3_datasets/human-chr14-synthetic_solution.txt