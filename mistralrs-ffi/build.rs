extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_tab_width(4)
        .with_line_length(100)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("mistralrs.h");
}
