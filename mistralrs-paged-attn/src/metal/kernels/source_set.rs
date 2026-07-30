mistralrs_metal_compile::metal_source_set! {
    pub const PAGED_ATTENTION_METAL_SOURCE_SET;
    library_name: "mistralrs_paged_attention",
    source_dir: "src/metal/kernels",
    metal_sources: [
        "copy_blocks",
        "pagedattention",
        "reshape_and_cache",
        "kv_scale_update",
        "gather_kv_cache",
    ],
    header_sources: ["utils", "float8"],
    include_only_sources: [],
}
