# How to use gprof and transform output into png

## Compilation

gprof is available via mingw but for some reason it does not work,
use wsl to compile, **remember the -pg flag!**

```wsl
gcc -pg -O0 main.c -o main
```

## Execution

run the program in wsl, if the compilation is correct you should see a gmon.out

## Profiling

```wsl
gprof main gmon.out > profiling.txt
```

## Generate graph

gprof2dot is available via pipx

```windows
gprof2dot -s -w profiling.txt | dot -Gdpi=300 -Tpng -o graph.png
```
