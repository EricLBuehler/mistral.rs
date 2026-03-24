/// Structured data extraction using `generate_structured<T>()`.
///
/// Demonstrates constraining the model's output to a JSON schema derived
/// from a Rust struct, then deserializing the result automatically.
///
/// Run with: `cargo run --release --example cookbook_structured -p mistralrs`
use anyhow::Result;
use mistralrs::{IsqBits, ModelBuilder, TextMessageRole, TextMessages};
use schemars::JsonSchema;
use serde::Deserialize;

/// A line item on an invoice.
#[derive(Debug, Deserialize, JsonSchema)]
struct LineItem {
    /// Description of the item.
    description: String,
    /// Quantity ordered.
    quantity: u32,
    /// Unit price in dollars.
    unit_price: f64,
}

/// Structured invoice data to extract from unstructured text.
#[derive(Debug, Deserialize, JsonSchema)]
struct Invoice {
    /// Vendor or company name.
    vendor: String,
    /// Invoice date in YYYY-MM-DD format.
    date: String,
    /// Line items on the invoice.
    items: Vec<LineItem>,
    /// Total amount in dollars.
    total: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-3-4b-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .build()
        .await?;

    let unstructured_text = r#"
        INVOICE from Acme Corp, dated 2025-01-15.
        Items:
        - 3x Widget A at $12.50 each
        - 1x Widget B at $45.00 each
        - 10x Bolt Pack at $3.25 each
        Total: $115.00
    "#;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        format!("Extract the invoice data from the following text:\n\n{unstructured_text}"),
    );

    let invoice: Invoice = model.generate_structured::<Invoice>(messages).await?;

    println!("Extracted Invoice:");
    println!("  Vendor: {}", invoice.vendor);
    println!("  Date:   {}", invoice.date);
    println!("  Items:");
    for item in &invoice.items {
        println!(
            "    - {}x {} @ ${:.2}",
            item.quantity, item.description, item.unit_price
        );
    }
    println!("  Total:  ${:.2}", invoice.total);

    Ok(())
}
