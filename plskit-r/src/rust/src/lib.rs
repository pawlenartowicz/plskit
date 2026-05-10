use extendr_api::prelude::*;

#[extendr]
fn version() -> &'static str {
    plskit::version()
}

extendr_module! {
    mod plskit;
    fn version;
}
