#define NOBDEF static inline
#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#define NOB_STRIP_PREFIX
#include "nob.h"

#define BUILD_FOLDER "build/"
#define SRC_FOLDER ""
#define INCLUDE_FOLDER "include/"

int main(int argc, char **argv) {
    GO_REBUILD_URSELF(argc, argv);
    if (!mkdir_if_not_exists(BUILD_FOLDER)) return 1;

    Cmd cmd = {0};

#if defined(PROFILE)
    cmd_append(&cmd, "clang-cl", "/Zi", "/Oy-");
    cmd_append(&cmd, "/I" INCLUDE_FOLDER);
    cmd_append(&cmd, "/openmp");
	cmd_append(&cmd, "/arch:AVX2");
#else
    cc(&cmd);
    cc_flags(&cmd);
    #ifndef _MSC_VER
		cmd_append(&cmd, "-O3");
		// cmd_append(&cmd, "-march=native");  // enable cpu specific optimization
		cmd_append(&cmd, "-I" INCLUDE_FOLDER);
		cmd_append(&cmd, "-fopenmp");
		cmd_append(&cmd, "-mavx2", "-mfma");
    #else
		cmd_append(&cmd, "/O3");
		cmd_append(&cmd, "/I" INCLUDE_FOLDER);
		cmd_append(&cmd, "/openmp");
		cmd_append(&cmd, "/arch:AVX2");
    #endif
#endif

    Cmd targets = {0};
    cc_inputs(&targets, SRC_FOLDER "main.c", INCLUDE_FOLDER "pffft.c");

    if (!compile_commands(&cmd, &targets, BUILD_FOLDER "compile_commands.json"))
        return 1;

    cc_output(&cmd, BUILD_FOLDER "main");
    cmd_extend(&cmd, &targets);
    if (!cmd_run(&cmd)) return 1;

    // cmd_append(&cmd, BUILD_FOLDER "main");
    // if (!cmd_run(&cmd)) return 1;
    return 0;
}
