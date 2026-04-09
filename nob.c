#define NOBDEF static inline
#define NOB_IMPLEMENTATION
#define NOB_WARN_DEPRECATED
#define NOB_STRIP_PREFIX
#include "nob.h"

#define BUILD_FOLDER "build/"
#define SRC_FOLDER ""
#define INCLUDE_FOLDER "include/"

typedef enum {
    COMPILER_GCC,
    COMPILER_CLANG,
    COMPILER_MSVC,
    COMPILER_CLANG_CL,
} Compiler;

typedef struct {
    Compiler compiler;
    bool debug;
    bool profiling;
    bool native;
    bool run;
} BuildConfig;

static void print_usage(const char *program) {
    fprintf(stderr, "Usage: %s [options]\n", program);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -gcc         Use gcc / MinGW (default)\n");
    fprintf(stderr, "  -clang       Use clang\n");
    fprintf(stderr, "  -msvc        Use cl (MSVC, requires Developer PowerShell)\n");
    fprintf(stderr, "  -clang-cl    Use clang-cl (for profiling with samply)\n");
    fprintf(stderr,
            "  -debug       Enable debug mode (defines DEBUG, disables "
            "optimizations)\n");
    fprintf(stderr, "  -profiling   Enable profiling flags (implies -clang-cl)\n");
    fprintf(stderr, "  -native      Enable -march=native (clang/gcc only)\n");
    fprintf(stderr, "  -run         Run the executable after building\n");
    fprintf(stderr, "  -help        Show this help message\n");
}

// clang (and clang-cl) on Windows needs _USE_MATH_DEFINES to expose M_PI etc.
// gcc exposes them by default via its own extensions — do not add the define.
// cl also gates M_PI behind _USE_MATH_DEFINES in its CRT headers.
static void apply_gnu_style_flags(Cmd *cmd, Compiler compiler, bool debug,
                                  bool native) {
    cc_flags(cmd);
    if (debug) {
        cmd_append(cmd, "-O0", "-g", "-DDEBUG");
    } else {
        cmd_append(cmd, "-O3");
        if (native) cmd_append(cmd, "-march=native");
    }
    if (compiler == COMPILER_CLANG) cmd_append(cmd, "-D_USE_MATH_DEFINES");
    cmd_append(cmd, "-fopenmp");
    cmd_append(cmd, "-mavx2", "-mfma");
    cmd_append(cmd, "-I" INCLUDE_FOLDER);
}

static void apply_msvc_style_flags(Cmd *cmd, Compiler compiler, bool debug,
                                   bool profiling) {
    nob_cmd_append(cmd, "/nologo");
    // nob_cmd_append(cmd, "/W4", "/D_CRT_SECURE_NO_WARNINGS");
    if (profiling) {
        cmd_append(cmd, "/Zi", "/Oy-");
    } else if (debug) {
        cmd_append(cmd, "/Zi", "/Od", "/DDEBUG");
    } else {
        cmd_append(cmd, "/O2");
    }
    cmd_append(cmd, "/D_USE_MATH_DEFINES");
    cmd_append(cmd, "/openmp");
    cmd_append(cmd, "/arch:AVX2");
    cmd_append(cmd, "/I" INCLUDE_FOLDER);
    (void)compiler;  // suppress unused var, may use in the future
}

// /Fe:<path> is the MSVC way to name the output executable.
// With multiple input files, cl.exe otherwise names the exe after the first .c file,
// which is fragile. /Fe: makes it explicit. Note: no space between /Fe: and the
// path.
static void cc_output_msvc(Cmd *cmd, const char *path) {
    String_Builder sb = {0};
    sb_append_cstr(&sb, "/Fe:");
    sb_append_cstr(&sb, path);
    sb_append_null(&sb);
    cmd_append(cmd, sb.items);
    // nob copies the string into its own storage, so sb can go out of scope safely.
}

// /Fo:<dir>\ tells cl where to put .obj files.
// The trailing backslash is required — without it cl treats it as a filename prefix.
static void cc_objdir_msvc(Cmd *cmd, const char *dir) {
    String_Builder sb = {0};
    sb_append_cstr(&sb, "/Fo:");
    sb_append_cstr(&sb, dir);
    // Ensure trailing backslash — MSVC requires it to treat this as a directory.
    if (dir[strlen(dir) - 1] != '\\' && dir[strlen(dir) - 1] != '/') {
        sb_append_cstr(&sb, "\\");
    }
    sb_append_null(&sb);
    cmd_append(cmd, sb.items);
}

static void cc_extra_outputs_msvc(Cmd *cmd, const char *dir) {
    String_Builder pdb = {0};
    sb_append_cstr(&pdb, "/Fd:");
    sb_append_cstr(&pdb, dir);
    sb_append_cstr(&pdb, "main.pdb");
    sb_append_null(&pdb);
    cmd_append(cmd, pdb.items);

    // cmd_append(cmd, "/link");
    // String_Builder ilk = {0};
    // sb_append_cstr(&ilk, "/ILK:");
    // sb_append_cstr(&ilk, dir);
    // sb_append_cstr(&ilk, "main.ilk");
    // sb_append_null(&ilk);
    // cmd_append(cmd, ilk.items);
}

// Check that a given executable is findable on PATH before we try to invoke it.
#ifdef _WIN32
static bool check_on_path(const char *exe) {
    char buf[MAX_PATH];
    return SearchPathA(NULL, exe, NULL, MAX_PATH, buf, NULL) != 0;
}
#else
static bool check_on_path(const char *exe) {
    (void)exe;
    return true;  // not needed outside Windows for these compilers
}
#endif

int main(int argc, char **argv) {
    GO_REBUILD_URSELF(argc, argv);

    const char *program = nob_shift(argv, argc);

    BuildConfig cfg = {
        .compiler = COMPILER_GCC,  // default
        .debug = false,
        .profiling = false,
        .native = false,
        .run = false,
    };

    while (argc > 0) {
        const char *flag = nob_shift(argv, argc);
        if (strcmp(flag, "-gcc") == 0) {
            cfg.compiler = COMPILER_GCC;
        } else if (strcmp(flag, "-clang") == 0) {
            cfg.compiler = COMPILER_CLANG;
        } else if (strcmp(flag, "-msvc") == 0) {
            cfg.compiler = COMPILER_MSVC;
        } else if (strcmp(flag, "-clang-cl") == 0) {
            cfg.compiler = COMPILER_CLANG_CL;
        } else if (strcmp(flag, "-debug") == 0) {
            cfg.debug = true;
        } else if (strcmp(flag, "-profiling") == 0) {
            cfg.profiling = true;
            cfg.compiler = COMPILER_CLANG_CL;
            nob_log(NOB_INFO, "-profiling flag forces clang-cl compiler");
        } else if (strcmp(flag, "-native") == 0) {
            cfg.native = true;
            if (cfg.compiler == COMPILER_CLANG_CL || cfg.compiler == COMPILER_MSVC)
                nob_log(NOB_WARNING,
                        "-march=native flag not available on msvc style compilers");
        } else if (strcmp(flag, "-run") == 0) {
            cfg.run = true;
        } else if (strcmp(flag, "-help") == 0 || strcmp(flag, "--help") == 0 ||
                   strcmp(flag, "-h") == 0) {
            print_usage(program);
            return 0;
        } else {
            nob_log(NOB_ERROR, "Unknown flag: %s", flag);
            print_usage(program);
            return 1;
        }
    }

    const char *compiler_names[] = {
        [COMPILER_GCC] = "gcc (MinGW)",
        [COMPILER_CLANG] = "clang",
        [COMPILER_MSVC] = "cl (MSVC)",
        [COMPILER_CLANG_CL] = "clang-cl",
    };
    nob_log(NOB_INFO, "Compiler : %s", compiler_names[cfg.compiler]);
    nob_log(NOB_INFO, "Debug    : %s", cfg.debug ? "yes" : "no");
    nob_log(NOB_INFO, "Profiling: %s", cfg.profiling ? "yes" : "no");
    // nob_log(NOB_INFO, "Native   : %s", cfg.native ? "yes" : "no");

    // Verify MSVC-style compilers are actually on PATH and give a helpful message if
    // not.
    if (cfg.compiler == COMPILER_MSVC) {
        if (!check_on_path("cl.exe")) {
            nob_log(NOB_ERROR, "'cl.exe' not found on PATH.");
            nob_log(NOB_ERROR,
                    "Open a Visual Studio Developer PowerShell or Developer Command "
                    "Prompt and try again.");
            return 1;
        }
    } else if (cfg.compiler == COMPILER_CLANG_CL) {
        if (!check_on_path("clang-cl.exe")) {
            nob_log(NOB_ERROR, "'clang-cl.exe' not found on PATH.");
            nob_log(
                NOB_ERROR,
                "Make sure LLVM is installed and its bin/ directory is on PATH.");
            return 1;
        }
    }

    if (!mkdir_if_not_exists(BUILD_FOLDER)) return 1;

    Cmd cmd = {0};
    bool msvc_style = false;

    switch (cfg.compiler) {
    case COMPILER_GCC:
        cmd_append(&cmd, "gcc");
        apply_gnu_style_flags(&cmd, COMPILER_GCC, cfg.debug, cfg.native);
        break;
    case COMPILER_CLANG:
        cmd_append(&cmd, "clang");
        apply_gnu_style_flags(&cmd, COMPILER_CLANG, cfg.debug, cfg.native);
        break;
    case COMPILER_MSVC:
        cmd_append(&cmd, "cl");
        apply_msvc_style_flags(&cmd, COMPILER_MSVC, cfg.debug, false);
        msvc_style = true;
        break;
    case COMPILER_CLANG_CL:
        cmd_append(&cmd, "clang-cl");
        apply_msvc_style_flags(&cmd, COMPILER_CLANG_CL, cfg.debug, cfg.profiling);
        msvc_style = true;
        break;
    }

    // NOTE: targets
    Cmd targets = {0};
    cc_inputs(&targets, SRC_FOLDER "main.c", INCLUDE_FOLDER "pffft.c");

    if (!compile_commands(&cmd, &targets, BUILD_FOLDER "compile_commands.json"))
        return 1;

    // NOTE: output
    if (msvc_style) {
        cc_output_msvc(&cmd, BUILD_FOLDER "main.exe");
        cc_objdir_msvc(&cmd, BUILD_FOLDER);
        cc_extra_outputs_msvc(&cmd, BUILD_FOLDER);
    } else {
        // cc_output is already smart enough to use the right flag based on the
        // available compilers in the current env (for example -o vs /Fe),
        // but given an env it will always pick the same flag.
        // By defining a separate cc_output_msvc func we can be more intentional
        cc_output(&cmd, BUILD_FOLDER "main.exe");
    }

    cmd_extend(&cmd, &targets);
    if (!cmd_run(&cmd)) return 1;

    if (cfg.run) {
        Cmd run_cmd = {0};
        cmd_append(&run_cmd, BUILD_FOLDER "main.exe");
        if (!cmd_run(&run_cmd)) return 1;
    }

    return 0;
}
