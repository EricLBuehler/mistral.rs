use mistralrs::{DecisionRequest, ModelBuilder};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = ModelBuilder::new("Contrastive-LM/CLM-v0.1-8B")
        .build()
        .await?;
    let request: DecisionRequest = serde_json::from_value(json!({
        "model": "default",
        "state": "My invoice was charged twice and nobody answers the phone!",
        "questions": {
            "urgent": {"type": "noul", "instructions": "Is this urgent?"},
            "team": {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
            "anger": {"type": "score", "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"]}
        }
    }))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&model.decide(request).await?)?
    );
    Ok(())
}
