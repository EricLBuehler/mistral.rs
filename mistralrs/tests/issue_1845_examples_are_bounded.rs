#[test]
fn isq_and_uqff_examples_set_sampler_max_len() {
    let isq = include_str!("../examples/isq/main.rs");
    let isq_bound_calls = isq.matches(".set_sampler_max_len(").count();
    assert_eq!(
        isq_bound_calls, 2,
        "isq example should cap both request call sites"
    );
    assert!(
        !isq.contains("TextMessages::new("),
        "isq example should use RequestBuilder for explicit bounded sampling"
    );

    let uqff = include_str!("../examples/uqff/main.rs");
    let uqff_bound_calls = uqff.matches(".set_sampler_max_len(").count();
    assert_eq!(
        uqff_bound_calls, 2,
        "uqff example should cap both request call sites"
    );
    assert!(
        !uqff.contains("TextMessages::new("),
        "uqff example should use RequestBuilder for explicit bounded sampling"
    );
}
