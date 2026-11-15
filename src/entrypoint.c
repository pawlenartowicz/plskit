// Forward routine registration to the Rust static library, so the linker
// keeps it, and install extendr's panic hook first.

void R_init_plskit_extendr(void *dll);
void register_extendr_panic_hook(void);

void R_init_plskit(void *dll) {
    register_extendr_panic_hook();
    R_init_plskit_extendr(dll);
}
