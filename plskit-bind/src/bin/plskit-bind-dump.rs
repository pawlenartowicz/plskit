//! Print the plskit-bind registry as JSON (for the R / Julia stub renderers
//! and `scripts/check-bind-registry.py`).

fn main() {
    print!("{}", plskit_bind::registry_json());
}
