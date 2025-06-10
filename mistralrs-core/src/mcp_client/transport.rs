use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;

/// Transport layer for MCP communication
#[async_trait::async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and receive a response
    async fn send_request(&self, method: &str, params: Value) -> Result<Value>;

    /// Check if the transport connection is healthy
    async fn ping(&self) -> Result<()>;

    /// Close the transport connection
    async fn close(&self) -> Result<()>;
}

/// HTTP-based MCP transport
pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    headers: HashMap<String, String>,
}

impl HttpTransport {
    pub fn new(
        base_url: String,
        timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let timeout = timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(30));
        let client = reqwest::Client::builder().timeout(timeout).build()?;

        Ok(Self {
            client,
            base_url,
            headers: headers.unwrap_or_default(),
        })
    }
}

#[async_trait::async_trait]
impl McpTransport for HttpTransport {
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });

        let mut request_builder = self.client.post(&self.base_url).json(&request_body);

        // Add custom headers
        for (key, value) in &self.headers {
            request_builder = request_builder.header(key, value);
        }

        let response = request_builder.send().await?;
        let response_body: Value = response.json().await?;

        // Check for JSON-RPC errors
        if let Some(error) = response_body.get("error") {
            return Err(anyhow::anyhow!("MCP server error: {}", error));
        }

        response_body
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No result in MCP response"))
    }

    async fn ping(&self) -> Result<()> {
        self.send_request("ping", Value::Null).await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // HTTP connections don't need explicit closing
        Ok(())
    }
}

/// Process-based MCP transport using stdin/stdout
pub struct ProcessTransport {
    child: std::sync::Arc<tokio::sync::Mutex<tokio::process::Child>>,
    stdin: std::sync::Arc<tokio::sync::Mutex<tokio::process::ChildStdin>>,
    stdout_reader:
        std::sync::Arc<tokio::sync::Mutex<tokio::io::BufReader<tokio::process::ChildStdout>>>,
}

impl ProcessTransport {
    pub async fn new(
        command: String,
        args: Vec<String>,
        work_dir: Option<String>,
        env: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        use tokio::io::BufReader;
        use tokio::process::Command;

        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        if let Some(dir) = work_dir {
            cmd.current_dir(dir);
        }

        if let Some(env_vars) = env {
            for (key, value) in env_vars {
                cmd.env(key, value);
            }
        }

        let mut child = cmd.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to get stdin handle"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to get stdout handle"))?;
        let stdout_reader = BufReader::new(stdout);

        Ok(Self {
            child: std::sync::Arc::new(tokio::sync::Mutex::new(child)),
            stdin: std::sync::Arc::new(tokio::sync::Mutex::new(stdin)),
            stdout_reader: std::sync::Arc::new(tokio::sync::Mutex::new(stdout_reader)),
        })
    }
}

#[async_trait::async_trait]
impl McpTransport for ProcessTransport {
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });

        // Send request via stdin
        let request_line = serde_json::to_string(&request_body)? + "\n";

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(request_line.as_bytes()).await?;
        stdin.flush().await?;
        drop(stdin);

        // Read response from stdout
        let mut stdout_reader = self.stdout_reader.lock().await;
        let mut response_line = String::new();
        stdout_reader.read_line(&mut response_line).await?;
        drop(stdout_reader);

        let response_body: Value = serde_json::from_str(&response_line.trim())?;

        // Check for JSON-RPC errors
        if let Some(error) = response_body.get("error") {
            return Err(anyhow::anyhow!("MCP server error: {}", error));
        }

        response_body
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No result in MCP response"))
    }

    async fn ping(&self) -> Result<()> {
        self.send_request("ping", Value::Null).await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        let mut child = self.child.lock().await;
        child.kill().await?;
        Ok(())
    }
}

#[allow(dead_code)]
/// WebSocket-based MCP transport
pub struct WebSocketTransport {
    // WebSocket implementation would go here
    // For now, this is a placeholder
    url: String,
    headers: HashMap<String, String>,
}

impl WebSocketTransport {
    pub async fn new(
        url: String,
        _timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        Ok(Self {
            url,
            headers: headers.unwrap_or_default(),
        })
    }
}

#[async_trait::async_trait]
impl McpTransport for WebSocketTransport {
    async fn send_request(&self, _method: &str, _params: Value) -> Result<Value> {
        // WebSocket implementation would go here
        // This is a placeholder for now
        // TODO
        Err(anyhow::anyhow!("WebSocket transport not yet implemented"))
    }

    async fn ping(&self) -> Result<()> {
        self.send_request("ping", Value::Null).await?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        // WebSocket cleanup would go here
        Ok(())
    }
}
