use anyhow::Result;
use mistralrs::{IsqType, MemoryUsage, RequestBuilder, TextMessageRole, TextModelBuilder};

const N_ITERS: u64 = 1000;
const BYTES_TO_MB: usize = 1024 * 1024;

const PROMPT: &str = r#"
The Rise of Rust: A New Era in Systems Programming
Introduction

Rust is a modern systems programming language that has garnered significant attention since its inception. Created by Graydon Hoare and backed by Mozilla Research, Rust has quickly risen to prominence due to its ability to balance performance, safety, and concurrency in ways that other languages struggle to achieve. While languages like C and C++ have long dominated systems programming, Rust offers a fresh approach by addressing many of the core issues that have historically plagued these languages—namely, memory safety and concurrency challenges.

In this essay, we will explore the key features of Rust, its advantages over other systems programming languages, and how it is shaping the future of software development, particularly in the realms of performance-critical and safe computing.

The Philosophy Behind Rust
Rust was designed to solve a key problem in systems programming: memory safety without sacrificing performance. Traditionally, languages like C and C++ have offered low-level access to memory, which is essential for writing efficient programs that interact closely with hardware. However, this power comes with significant risks, especially when it comes to bugs such as buffer overflows, null pointer dereferencing, and use-after-free errors. These types of bugs not only cause crashes but also open up security vulnerabilities, which have become a major issue in modern software development.

Rust's approach to memory safety is built on a few key principles:

Ownership and Borrowing: Rust introduces a unique ownership system that ensures memory safety at compile time. Each value in Rust has a single owner, and when the owner goes out of scope, the value is deallocated. This ensures that memory leaks and dangling pointers are virtually impossible. Furthermore, Rust's borrowing system allows references to be shared, but only in ways that are provably safe. For example, mutable references cannot be aliased, preventing many common concurrency issues.

Zero-Cost Abstractions: Rust provides high-level abstractions such as iterators and closures without incurring runtime penalties. This is crucial in systems programming, where performance is paramount. Unlike languages that rely on garbage collection (like Java or Go), Rustabsmins memory model allows developers to write high-performance code while still benefiting from the safety of modern abstractions.
"#;

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q4K)
        .with_prefix_cache_n(None)
        .with_logging()
        // .with_paged_attn(|| mistralrs::PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    for i in 0..N_ITERS {
        let messages = RequestBuilder::new()
            .add_message(
                TextMessageRole::User,
                PROMPT,
            )
            .set_sampler_max_len(1000);

        println!("Sending request {}...", i + 1);
        let response = model.send_chat_request(messages).await?;

        let amount = MemoryUsage.get_memory_available(&model.config().device)? / BYTES_TO_MB;

        println!("{amount}");
        println!("{}", response.usage.total_time_sec);
        println!("{:?}", response.choices[0].message.content);
    }

    Ok(())
}
