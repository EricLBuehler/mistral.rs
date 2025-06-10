use anyhow::Result;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use http::{Request, Uri};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};

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

    /// Parse Server-Sent Events response to extract JSON-RPC message
    fn parse_sse_response(sse_text: &str) -> Result<Value> {
        // SSE format: data: <json>\n\n or event: <type>\ndata: <json>\n\n
        let mut json_data = None;

        for line in sse_text.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            // Parse SSE field
            if let Some((field, value)) = line.split_once(':') {
                let field = field.trim();
                let value = value.trim();

                match field {
                    "data" => {
                        // Try to parse the JSON data
                        if let Ok(parsed) = serde_json::from_str::<Value>(value) {
                            json_data = Some(parsed);
                            break;
                        }
                    }
                    "event" => {
                        // Handle different event types if needed
                        continue;
                    }
                    _ => {
                        // Ignore other SSE fields like id, retry, etc.
                        continue;
                    }
                }
            }
        }

        json_data.ok_or_else(|| anyhow::anyhow!("No valid JSON data found in SSE response"))
    }
}

#[async_trait::async_trait]
impl McpTransport for HttpTransport {
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        // Ensure params is an object, not null
        let params = if params.is_null() {
            serde_json::json!({})
        } else {
            params
        };

        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });

        let mut request_builder = self
            .client
            .post(&self.base_url)
            .json(&request_body)
            .header("Accept", "application/json, text/event-stream");

        // Add custom headers
        for (key, value) in &self.headers {
            request_builder = request_builder.header(key, value);
        }

        let response = request_builder.send().await?;

        // Check content type and handle accordingly
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        let response_body: Value = if content_type.contains("text/event-stream") {
            // Handle Server-Sent Events
            let response_text = response.text().await?;
            Self::parse_sse_response(&response_text)?
        } else {
            // Handle regular JSON response
            response.json().await?
        };

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

        // Ensure params is an object, not null
        let params = if params.is_null() {
            serde_json::json!({})
        } else {
            params
        };

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

/// WebSocket-based MCP transport
pub struct WebSocketTransport {
    write: std::sync::Arc<
        tokio::sync::Mutex<SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>>,
    >,
    read:
        std::sync::Arc<tokio::sync::Mutex<SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    request_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl WebSocketTransport {
    pub async fn new(
        url: String,
        _timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        // Create request with headers
        let uri: Uri = url
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid WebSocket URL: {}", e))?;
        let mut request = Request::builder()
            .uri(uri)
            .body(())
            .map_err(|e| anyhow::anyhow!("Failed to create WebSocket request: {}", e))?;

        // Add headers if provided
        if let Some(headers) = headers {
            let req_headers = request.headers_mut();
            for (key, value) in headers {
                let header_name = key
                    .parse::<http::header::HeaderName>()
                    .map_err(|e| anyhow::anyhow!("Invalid header key: {}", e))?;
                let header_value = value
                    .parse::<http::header::HeaderValue>()
                    .map_err(|e| anyhow::anyhow!("Invalid header value: {}", e))?;
                req_headers.insert(header_name, header_value);
            }
        }

        // Connect to WebSocket
        let (ws_stream, _) = connect_async(request)
            .await
            .map_err(|e| anyhow::anyhow!("WebSocket connection failed: {}", e))?;

        // Split the stream
        let (write, read) = ws_stream.split();

        Ok(Self {
            write: std::sync::Arc::new(tokio::sync::Mutex::new(write)),
            read: std::sync::Arc::new(tokio::sync::Mutex::new(read)),
            request_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
        })
    }
}

#[async_trait::async_trait]
impl McpTransport for WebSocketTransport {
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        // Ensure params is an object, not null
        let params = if params.is_null() {
            serde_json::json!({})
        } else {
            params
        };

        // Generate unique request ID
        let id = self
            .request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        // Send request
        let message = Message::Text(serde_json::to_string(&request_body)?);

        {
            let mut write = self.write.lock().await;
            write
                .send(message)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send WebSocket message: {}", e))?;
        }

        // Read response
        loop {
            let mut read = self.read.lock().await;
            let msg = read
                .next()
                .await
                .ok_or_else(|| anyhow::anyhow!("WebSocket connection closed"))?
                .map_err(|e| anyhow::anyhow!("WebSocket read error: {}", e))?;
            drop(read);

            match msg {
                Message::Text(text) => {
                    let response_body: Value = serde_json::from_str(&text)?;

                    // Check if this is the response to our request
                    if let Some(response_id) = response_body.get("id").and_then(|v| v.as_u64()) {
                        if response_id == id {
                            // Check for JSON-RPC errors
                            if let Some(error) = response_body.get("error") {
                                return Err(anyhow::anyhow!("MCP server error: {}", error));
                            }

                            return response_body
                                .get("result")
                                .cloned()
                                .ok_or_else(|| anyhow::anyhow!("No result in MCP response"));
                        }
                    }
                    // If it's not our response, continue reading
                }
                Message::Binary(_) => {
                    // Handle binary messages if needed, for now skip
                    continue;
                }
                Message::Close(_) => {
                    return Err(anyhow::anyhow!("WebSocket connection closed by server"));
                }
                Message::Ping(_) | Message::Pong(_) => {
                    // Handle ping/pong frames, continue reading
                    continue;
                }
                Message::Frame(_) => {
                    // Raw frames, continue reading
                    continue;
                }
            }
        }
    }

    async fn ping(&self) -> Result<()> {
        let ping_message = Message::Ping(vec![]);
        let mut write = self.write.lock().await;
        write
            .send(ping_message)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send ping: {}", e))?;
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        let close_message = Message::Close(None);
        let mut write = self.write.lock().await;
        write
            .send(close_message)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send close message: {}", e))?;
        Ok(())
    }
}
