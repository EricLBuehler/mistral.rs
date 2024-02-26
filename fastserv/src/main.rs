use axum::{routing::get, Router};
use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    /// Port to serve on.
    #[arg(short, long)]
    port: String,
}

async fn root() -> &'static str {
    "Hello World!"
}

fn get_router() -> Router {
    Router::new().route("/", get(root))
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let app = get_router();

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", args.port))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
