mistralrs_metal_compile::metal_source_set! {
    pub const GDN_METAL_SOURCE_SET;
    library_name: "mistralrs_gdn",
    source_dir: "src/metal/kernels",
    metal_sources: ["gdn"],
    header_sources: [],
    include_only_sources: [],
}

mistralrs_metal_compile::metal_source_set! {
    pub const SSM_METAL_SOURCE_SET;
    library_name: "mistralrs_ssm",
    source_dir: "src/metal/kernels",
    metal_sources: ["ssm"],
    header_sources: [],
    include_only_sources: [],
}
