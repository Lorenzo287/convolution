#ifndef CUSTOM_MAIN_H
#define CUSTOM_MAIN_H

#define STB_LEAKCHECK_IMPLEMENTATION
#include "stb_leakcheck.h"

int custom_main(int argc, char **argv);

#define main(...)                       \
    main(int argc, char **argv) {       \
        atexit(stb_leakcheck_dumpmem);  \
        return custom_main(argc, argv); \
    }                                   \
    int custom_main(__VA_ARGS__)

#endif  // CUSTOM_MAIN_H
