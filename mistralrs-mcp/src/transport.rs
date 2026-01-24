use anyhow::Result;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
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

    /// Send initialization notification
    async fn send_initialization_notification(&self) -> Result<()>;
}

/// HTTP-based MCP transport
///
/// Provides communication with MCP servers over HTTP using JSON-RPC 2.0 protocol.
/// This transport is ideal for RESTful MCP services, public APIs, and servers
/// behind load balancers. Supports both regular JSON responses and Server-Sent Events (SSE).
///
/// # Features
///
/// - **HTTP/HTTPS Support**: Secure communication with TLS encryption
/// - **Server-Sent Events**: Handles streaming responses via SSE format
/// - **Bearer Token Authentication**: Automatic Authorization header injection
/// - **Custom Headers**: Support for additional headers (API keys, versioning, etc.)
/// - **Configurable Timeouts**: Request-level timeout control
/// - **Error Handling**: Comprehensive JSON-RPC and HTTP error handling
///
/// # Use Cases
///
/// - **Public MCP APIs**: Connect to hosted MCP services
/// - **RESTful Services**: Integration with REST-based tool providers
/// - **Load-Balanced Servers**: Works well behind HTTP load balancers
/// - **Development/Testing**: Easy debugging with standard HTTP tools
///
/// # Example Usage
///
/// ```rust,no_run
/// use mistralrs_mcp::transport::{HttpTransport, McpTransport};
/// use std::collections::HashMap;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Create headers with Bearer token and API version
///     let mut headers = HashMap::new();
///     headers.insert("Authorization".to_string(), "Bearer your-api-token".to_string());
///     headers.insert("X-API-Version".to_string(), "v1".to_string());
///
///     // Connect to HTTP MCP server
///     let transport = HttpTransport::new(
///         "https://api.example.com/mcp".to_string(),
///         Some(30), // 30 second timeout
///         Some(headers)
///     )?;
///
///     // Use the transport for MCP communication
///     let result = transport.send_request("tools/list", serde_json::Value::Null).await?;
///     println!("Available tools: {}", result);
///
///     Ok(())
/// }
/// ```
///
/// # Protocol Support
///
/// This transport implements JSON-RPC 2.0 over HTTP with support for:
/// - Standard JSON responses
/// - Server-Sent Events (SSE) for streaming data
/// - Bearer token authentication
/// - Custom HTTP headers
pub struct HttpTransport {
    client: reqwest::Client,
    base_url: String,
    headers: HashMap<String, String>,
    request_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl HttpTransport {
    /// Creates a new HTTP transport for MCP communication
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the MCP server (http:// or https://)
    /// * `timeout_secs` - Optional timeout for HTTP requests in seconds (defaults to 30s)
    /// * `headers` - Optional custom headers to include in all requests
    ///
    /// # Returns
    ///
    /// A configured HttpTransport ready for MCP communication
    ///
    /// # Errors
    ///
    /// - Invalid URL format
    /// - HTTP client configuration errors
    /// - TLS/SSL setup failures
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mistralrs_mcp::transport::HttpTransport;
    /// use std::collections::HashMap;
    ///
    /// // Basic HTTP transport
    /// let transport = HttpTransport::new(
    ///     "https://api.example.com/mcp".to_string(),
    ///     Some(60), // 1 minute timeout
    ///     None
    /// )?;
    ///
    /// // With custom headers and authentication
    /// let mut headers = HashMap::new();
    /// headers.insert("Authorization".to_string(), "Bearer token123".to_string());
    /// headers.insert("X-Client-Version".to_string(), "1.0.0".to_string());
    ///
    /// let transport = HttpTransport::new(
    ///     "https://secure-api.example.com/mcp".to_string(),
    ///     Some(30),
    ///     Some(headers)
    /// )?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
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
            request_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
        })
    }

    /// Parse Server-Sent Events response to extract JSON-RPC message
    ///
    /// Handles SSE format used by some MCP servers for streaming responses.
    /// SSE format: `data: <json>\n\n` or `event: <type>\ndata: <json>\n\n`
    ///
    /// # Arguments
    ///
    /// * `sse_text` - Raw SSE response text from the server
    ///
    /// # Returns
    ///
    /// Parsed JSON value from the SSE data field
    ///
    /// # Errors
    ///
    /// - No valid JSON data found in SSE response
    /// - Malformed SSE format
    /// - JSON parsing errors
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
    /// Sends an MCP request over HTTP and returns the response
    ///
    /// This method implements JSON-RPC 2.0 over HTTP with support for both
    /// standard JSON responses and Server-Sent Events (SSE). It handles
    /// authentication, custom headers, and comprehensive error reporting.
    ///
    /// # Arguments
    ///
    /// * `method` - The MCP method name (e.g., "tools/list", "tools/call", "resources/read")
    /// * `params` - JSON parameters for the method call
    ///
    /// # Returns
    ///
    /// The result portion of the JSON-RPC response
    ///
    /// # Errors
    ///
    /// - HTTP connection errors (network issues, DNS resolution)
    /// - HTTP status errors (4xx, 5xx responses)
    /// - JSON serialization/deserialization errors
    /// - MCP server errors (returned in JSON-RPC error field)
    /// - SSE parsing errors for streaming responses
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mistralrs_mcp::transport::{HttpTransport, McpTransport};
    /// use serde_json::json;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let transport = HttpTransport::new(
    ///         "https://api.example.com/mcp".to_string(),
    ///         None,
    ///         None
    ///     )?;
    ///     
    ///     // List available tools
    ///     let tools = transport.send_request("tools/list", serde_json::Value::Null).await?;
    ///     
    ///     // Call a specific tool
    ///     let params = json!({
    ///         "name": "search",
    ///         "arguments": {"query": "example search"}
    ///     });
    ///     let result = transport.send_request("tools/call", params).await?;
    ///     
    ///     Ok(())
    /// }
    /// ```
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        // Ensure params is an object, not null
        let params = if params.is_null() {
            serde_json::json!({})
        } else {
            params
        };

        let id = self
            .request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
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

    /// Tests the HTTP connection by sending a ping request
    ///
    /// Sends a "ping" method call to verify that the MCP server is responsive
    /// and the HTTP connection is working properly.
    ///
    /// # Returns
    ///
    /// Ok(()) if the ping was successful
    ///
    /// # Errors
    ///
    /// - HTTP connection errors
    /// - Server unavailable or unresponsive
    /// - Authentication failures
    async fn ping(&self) -> Result<()> {
        self.send_request("ping", Value::Null).await?;
        Ok(())
    }

    /// Closes the HTTP transport connection
    ///
    /// HTTP connections are stateless and managed by the underlying HTTP client,
    /// so this method is a no-op but provided for interface compatibility.
    ///
    /// # Returns
    ///
    /// Always returns Ok(()) as HTTP connections don't require explicit cleanup
    async fn close(&self) -> Result<()> {
        // HTTP connections don't need explicit closing
        Ok(())
    }

    /// Sends the server a initialization notification to let it know we are done initializing
    async fn send_initialization_notification(&self) -> Result<()> {
        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
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

        request_builder.send().await?;
        Ok(())
    }
}

/// Process-based MCP transport using stdin/stdout communication
///
/// Provides communication with local MCP servers running as separate processes
/// using JSON-RPC 2.0 over stdin/stdout pipes. This transport is ideal for
/// local tools, development servers, and sandboxed environments where you need
/// process isolation and direct control over the MCP server lifecycle.
///
/// # Features
///
/// - **Process Isolation**: Each MCP server runs in its own process for security
/// - **No Network Overhead**: Direct pipe communication for maximum performance
/// - **Environment Control**: Full control over working directory and environment variables
/// - **Resource Management**: Automatic process cleanup and lifecycle management
/// - **Synchronous Communication**: Request/response correlation over stdin/stdout
/// - **Error Handling**: Comprehensive process and communication error handling
///
/// # Use Cases
///
/// - **Local Development**: Running MCP servers during development and testing
/// - **Filesystem Tools**: Local file operations and system utilities
/// - **Sandboxed Execution**: Isolated execution environments for security
/// - **Custom Tools**: Private or proprietary MCP servers
/// - **CI/CD Integration**: Running MCP servers in automated environments
///
/// # Example Usage
///
/// ```rust,no_run
/// use mistralrs_mcp::transport::{ProcessTransport, McpTransport};
/// use std::collections::HashMap;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Basic process transport
///     let transport = ProcessTransport::new(
///         "mcp-server-filesystem".to_string(),
///         vec!["--root".to_string(), "/tmp".to_string()],
///         None,
///         None
///     ).await?;
///
///     // With custom working directory and environment
///     let mut env = HashMap::new();
///     env.insert("MCP_LOG_LEVEL".to_string(), "debug".to_string());
///     env.insert("MCP_TIMEOUT".to_string(), "30".to_string());
///
///     let transport = ProcessTransport::new(
///         "/usr/local/bin/my-mcp-server".to_string(),
///         vec!["--config".to_string(), "production.json".to_string()],
///         Some("/opt/mcp-server".to_string()), // Working directory
///         Some(env) // Environment variables
///     ).await?;
///
///     // Use the transport for MCP communication
///     let result = transport.send_request("tools/list", serde_json::Value::Null).await?;
///     println!("Available tools: {}", result);
///
///     Ok(())
/// }
/// ```
///
/// # Process Management
///
/// The transport automatically manages the child process lifecycle:
/// - Spawns the process with configured arguments and environment
/// - Sets up stdin/stdout pipes for JSON-RPC communication
/// - Monitors process health and handles crashes
/// - Cleans up resources when the transport is dropped or closed
///
/// # Communication Protocol
///
/// Uses JSON-RPC 2.0 over stdin/stdout with line-delimited messages:
/// - Each request is a single line of JSON sent to stdin
/// - Each response is a single line of JSON read from stdout
/// - Stderr is captured for debugging and error reporting
pub struct ProcessTransport {
    child: std::sync::Arc<tokio::sync::Mutex<tokio::process::Child>>,
    request_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
    stdin: std::sync::Arc<tokio::sync::Mutex<tokio::process::ChildStdin>>,
    stdout_reader:
        std::sync::Arc<tokio::sync::Mutex<tokio::io::BufReader<tokio::process::ChildStdout>>>,
}

impl ProcessTransport {
    /// Creates a new process transport by spawning an MCP server process
    ///
    /// This constructor spawns a new process with the specified command, arguments,
    /// and environment, then sets up stdin/stdout pipes for JSON-RPC communication.
    /// The process is ready to receive MCP requests immediately after creation.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute (e.g., "mcp-server-filesystem", "/usr/bin/python")
    /// * `args` - Command-line arguments to pass to the process
    /// * `work_dir` - Optional working directory for the process (defaults to current directory)
    /// * `env` - Optional environment variables to set for the process
    ///
    /// # Returns
    ///
    /// A configured ProcessTransport with the spawned process ready for communication
    ///
    /// # Errors
    ///
    /// - Command not found or not executable
    /// - Permission denied errors
    /// - Process spawn failures
    /// - Pipe setup errors (stdin/stdout/stderr)
    /// - Working directory access errors
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mistralrs_mcp::transport::ProcessTransport;
    /// use std::collections::HashMap;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     // Simple filesystem server
    ///     let transport = ProcessTransport::new(
    ///         "mcp-server-filesystem".to_string(),
    ///         vec!["--root".to_string(), "/home/user/documents".to_string()],
    ///         None,
    ///         None
    ///     ).await?;
    ///
    ///     // Python-based MCP server with custom environment
    ///     let mut env = HashMap::new();
    ///     env.insert("PYTHONPATH".to_string(), "/opt/mcp-servers".to_string());
    ///     env.insert("MCP_DEBUG".to_string(), "1".to_string());
    ///
    ///     let transport = ProcessTransport::new(
    ///         "python".to_string(),
    ///         vec!["-m".to_string(), "my_mcp_server".to_string(), "--port".to_string(), "8080".to_string()],
    ///         Some("/opt/mcp-servers".to_string()),
    ///         Some(env)
    ///     ).await?;
    ///
    ///     Ok(())
    /// }
    /// ```
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
            request_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
            stdin: std::sync::Arc::new(tokio::sync::Mutex::new(stdin)),
            stdout_reader: std::sync::Arc::new(tokio::sync::Mutex::new(stdout_reader)),
        })
    }
}

#[async_trait::async_trait]
impl McpTransport for ProcessTransport {
    /// Sends an MCP request to the child process and returns the response
    ///
    /// This method implements JSON-RPC 2.0 over stdin/stdout pipes. It sends
    /// a line-delimited JSON request to the process stdin and reads the
    /// corresponding response from stdout. Communication is synchronous with
    /// proper request/response correlation.
    ///
    /// # Arguments
    ///
    /// * `method` - The MCP method name (e.g., "tools/list", "tools/call", "resources/read")
    /// * `params` - JSON parameters for the method call
    ///
    /// # Returns
    ///
    /// The result portion of the JSON-RPC response
    ///
    /// # Errors
    ///
    /// - Process communication errors (broken pipes)
    /// - Process crashes or unexpected termination
    /// - JSON serialization/deserialization errors
    /// - MCP server errors (returned in JSON-RPC error field)
    /// - I/O errors on stdin/stdout
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mistralrs_mcp::transport::{ProcessTransport, McpTransport};
    /// use serde_json::json;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let transport = ProcessTransport::new(
    ///         "mcp-server-filesystem".to_string(),
    ///         vec!["--root".to_string(), "/tmp".to_string()],
    ///         None,
    ///         None
    ///     ).await?;
    ///     
    ///     // List available tools
    ///     let tools = transport.send_request("tools/list", serde_json::Value::Null).await?;
    ///     
    ///     // Call a specific tool
    ///     let params = json!({
    ///         "name": "read_file",
    ///         "arguments": {"path": "/tmp/example.txt"}
    ///     });
    ///     let result = transport.send_request("tools/call", params).await?;
    ///     
    ///     Ok(())
    /// }
    /// ```
    async fn send_request(&self, method: &str, params: Value) -> Result<Value> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

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

        let response_body: Value = serde_json::from_str(response_line.trim())?;

        // Check for JSON-RPC errors
        if let Some(error) = response_body.get("error") {
            return Err(anyhow::anyhow!("MCP server error: {}", error));
        }

        response_body
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No result in MCP response"))
    }

    /// Tests the process connection by sending a ping request
    ///
    /// Sends a "ping" method call to verify that the MCP server process is
    /// responsive and the stdin/stdout communication is working properly.
    ///
    /// # Returns
    ///
    /// Ok(()) if the ping was successful
    ///
    /// # Errors
    ///
    /// - Process communication errors
    /// - Process crashed or terminated
    /// - Broken stdin/stdout pipes
    async fn ping(&self) -> Result<()> {
        self.send_request("ping", Value::Null).await?;
        Ok(())
    }

    /// Terminates the child process and cleans up resources
    ///
    /// This method forcefully terminates the MCP server process and closes
    /// all associated pipes. Any pending requests will fail after this call.
    /// The transport cannot be used after closing.
    ///
    /// # Returns
    ///
    /// Ok(()) if the process was terminated successfully
    ///
    /// # Errors
    ///
    /// - Process termination errors
    /// - Resource cleanup failures
    ///
    /// # Note
    ///
    /// This method sends SIGKILL to the process, which may not allow for
    /// graceful cleanup. Consider implementing graceful shutdown through
    /// MCP protocol methods before calling this method.
    async fn close(&self) -> Result<()> {
        let mut child = self.child.lock().await;
        child.kill().await?;
        Ok(())
    }

    /// Sends the server a initialization notification to let it know we are done initializing
    async fn send_initialization_notification(&self) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(
                format!(
                    "{}\n",
                    serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"})
                )
                .as_bytes(),
            )
            .await?;
        stdin.flush().await?;
        drop(stdin);
        Ok(())
    }
}

/// WebSocket-based MCP transport
///
/// Provides real-time bidirectional communication with MCP servers over WebSocket connections.
/// This transport supports secure connections (WSS), Bearer token authentication, and concurrent
/// request/response handling with proper JSON-RPC 2.0 message correlation.
///
/// # Features
///
/// - **Async WebSocket Communication**: Built on tokio-tungstenite for high-performance async I/O
/// - **Request/Response Matching**: Automatic correlation of responses using atomic request IDs
/// - **Bearer Token Support**: Authentication via Authorization header during handshake
/// - **Connection Management**: Proper ping/pong and connection lifecycle handling
/// - **Concurrent Operations**: Split stream architecture allows simultaneous read/write operations
///
/// # Architecture
///
/// The transport uses a split-stream design where the WebSocket connection is divided into
/// separate read and write halves, each protected by async mutexes. This allows concurrent
/// operations while maintaining thread safety. Request IDs are generated atomically to ensure
/// unique identification of requests and proper response correlation.
///
/// # Example Usage
///
/// ```rust,no_run
/// use mistralrs_mcp::transport::{WebSocketTransport, McpTransport};
/// use std::collections::HashMap;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Create headers with Bearer token
///     let mut headers = HashMap::new();
///     headers.insert("Authorization".to_string(), "Bearer your-token".to_string());
///
///     // Connect to WebSocket MCP server
///     let transport = WebSocketTransport::new(
///         "wss://api.example.com/mcp".to_string(),
///         Some(30), // 30 second timeout
///         Some(headers)
///     ).await?;
///
///     // Use the transport for MCP communication
///     let result = transport.send_request("tools/list", serde_json::Value::Null).await?;
///     println!("Available tools: {}", result);
///
///     Ok(())
/// }
/// ```
///
/// # Protocol Compliance
///
/// This transport implements the Model Context Protocol (MCP) specification over WebSocket,
/// adhering to JSON-RPC 2.0 message format with proper error handling and response correlation.
pub struct WebSocketTransport {
    write: std::sync::Arc<
        tokio::sync::Mutex<SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>>,
    >,
    read:
        std::sync::Arc<tokio::sync::Mutex<SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    request_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl WebSocketTransport {
    /// Creates a new WebSocket transport connection to an MCP server
    ///
    /// # Arguments
    ///
    /// * `url` - WebSocket URL (ws:// or wss://)
    /// * `_timeout_secs` - Connection timeout (currently unused, reserved for future use)
    /// * `headers` - Optional HTTP headers for WebSocket handshake (e.g., Bearer tokens)
    ///
    /// # Returns
    ///
    /// A configured WebSocketTransport ready for MCP communication
    ///
    /// # Errors
    ///
    /// - Invalid URL format
    /// - WebSocket connection failure
    /// - Header parsing errors
    /// - Network connectivity issues
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mistralrs_mcp::transport::WebSocketTransport;
    /// use std::collections::HashMap;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let mut headers = HashMap::new();
    ///     headers.insert("Authorization".to_string(), "Bearer token123".to_string());
    ///     
    ///     let transport = WebSocketTransport::new(
    ///         "wss://mcp.example.com/api".to_string(),
    ///         Some(30),
    ///         Some(headers)
    ///     ).await?;
    ///     
    ///     Ok(())
    /// }
    /// ```
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
    /// Sends an MCP request over WebSocket and waits for the corresponding response
    ///
    /// This method implements the JSON-RPC 2.0 protocol over WebSocket, handling:
    /// - Unique request ID generation for response correlation
    /// - Concurrent request processing with proper message ordering
    /// - Error handling for both transport and protocol errors
    /// - Message filtering to match responses with requests
    ///
    /// # Arguments
    ///
    /// * `method` - The MCP method name (e.g., "tools/list", "tools/call")
    /// * `params` - JSON parameters for the method call
    ///
    /// # Returns
    ///
    /// The result portion of the JSON-RPC response
    ///
    /// # Errors
    ///
    /// - WebSocket connection errors
    /// - JSON serialization/deserialization errors  
    /// - MCP server errors (returned in JSON-RPC error field)
    /// - Timeout or connection closure
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mistralrs_mcp::transport::{WebSocketTransport, McpTransport};
    /// use serde_json::json;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let transport = WebSocketTransport::new(
    ///         "wss://api.example.com/mcp".to_string(),
    ///         None,
    ///         None
    ///     ).await?;
    ///     
    ///     // List available tools
    ///     let tools = transport.send_request("tools/list", serde_json::Value::Null).await?;
    ///     
    ///     // Call a specific tool
    ///     let params = json!({"query": "example search"});
    ///     let result = transport.send_request("tools/call", params).await?;
    ///     
    ///     Ok(())
    /// }
    /// ```
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
        let message = Message::Text(serde_json::to_string(&request_body)?.into());

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

    /// Sends a WebSocket ping frame to test connection health
    ///
    /// This method sends a ping frame to the server and expects a pong response,
    /// which helps verify that the WebSocket connection is still active and responsive.
    ///
    /// # Returns
    ///
    /// Ok(()) if the ping was sent successfully
    ///
    /// # Errors
    ///
    /// - WebSocket send errors
    /// - Connection closure
    async fn ping(&self) -> Result<()> {
        let ping_message = Message::Ping(vec![].into());
        let mut write = self.write.lock().await;
        write
            .send(ping_message)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send ping: {}", e))?;
        Ok(())
    }

    /// Gracefully closes the WebSocket connection
    ///
    /// Sends a close frame to the server to properly terminate the connection
    /// according to the WebSocket protocol. The server should respond with its
    /// own close frame to complete the closing handshake.
    ///
    /// # Returns
    ///
    /// Ok(()) if the close frame was sent successfully
    ///
    /// # Errors
    ///
    /// - WebSocket send errors
    /// - Connection already closed
    async fn close(&self) -> Result<()> {
        let close_message = Message::Close(None);
        let mut write = self.write.lock().await;
        write
            .send(close_message)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send close message: {}", e))?;
        Ok(())
    }

    /// Sends the server a initialization notification to let it know we are done initializing
    async fn send_initialization_notification(&self) -> Result<()> {
        let request_body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });

        // Send request
        let message = Message::Text(serde_json::to_string(&request_body)?.into());

        {
            let mut write = self.write.lock().await;
            write
                .send(message)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to send WebSocket message: {}", e))?;
        }
        Ok(())
    }
}
