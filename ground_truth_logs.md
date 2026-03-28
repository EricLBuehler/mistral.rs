# Issue #1947: WebSocketTransport Concurrency Fix

## [The Expectation]
The `WebSocketTransport` should correctly handle concurrent `send_request` calls. Each request should receive its own response based on the JSON-RPC `id` field. Multiple concurrent requests should be able to share a single WebSocket connection without stealing each other's responses.

## [Before Fix Terminal Proof]
By inspection of `mistralrs-mcp/src/transport.rs`, the `send_request` implementation used a competitive read-lock loop:

```rust
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
                            // ... return result ...
                        }
                    }
                    // If it's not our response, continue reading
                }
                // ...
            }
        }
```

In this implementation, if Request A and Request B are made concurrently, Request A might lock the reader, receive Response B, see that `response_id (B) != id (A)`, and continue the loop. Response B is then lost for Request B, which will eventually time out or fail.

## [The Applied Code Diff]
```diff
--- a/mistralrs-mcp/src/transport.rs
+++ b/mistralrs-mcp/src/transport.rs
@@ -1,11 +1,13 @@
 use anyhow::Result;
-use futures_util::stream::{SplitSink, SplitStream};
+use futures_util::stream::SplitSink;
 use futures_util::{SinkExt, StreamExt};
 use http::{Request, Uri};
 use serde_json::Value;
 use std::collections::HashMap;
+use std::sync::Arc;
 use std::time::Duration;
 use tokio::net::TcpStream;
+use tokio::sync::{oneshot, Mutex};
 use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};
 
 /// Transport layer for MCP communication
@@ -775,12 +777,10 @@
 /// This transport implements the Model Context Protocol (MCP) specification over WebSocket,
 /// adhering to JSON-RPC 2.0 message format with proper error handling and response correlation.
 pub struct WebSocketTransport {
-    write: std::sync::Arc<
-        tokio::sync::Mutex<SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>>,
-    >,
-    read:
-        std::sync::Arc<tokio::sync::Mutex<SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
-    request_id: std::sync::Arc<std::sync::atomic::AtomicU64>,
+    write: Arc<Mutex<SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>>>,
+    pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>>,
+    request_id: Arc<std::sync::atomic::AtomicU64>,
+    timeout: Duration,
 }
 
 impl WebSocketTransport {
@@ -789,7 +789,7 @@
     /// # Arguments
     ///
     /// * `url` - WebSocket URL (ws:// or wss://)
-    /// * `_timeout_secs` - Connection timeout (currently unused, reserved for future use)
+    /// * `timeout_secs` - Connection timeout
     /// * `headers` - Optional HTTP headers for WebSocket handshake (e.g., Bearer tokens)
     ///
     /// # Returns
@@ -825,7 +825,7 @@
     /// ```
     pub async fn new(
         url: String,
-        _timeout_secs: Option<u64>,
+        timeout_secs: Option<u64>,
         headers: Option<HashMap<String, String>>,
     ) -> Result<Self> {
         // Create request with headers
@@ -857,12 +857,52 @@
             .map_err(|e| anyhow::anyhow!("WebSocket connection failed: {}", e))?;
 
         // Split the stream
-        let (write, read) = ws_stream.split();
+        let (write, mut read) = ws_stream.split();
+
+        let pending_requests: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value>>>>> =
+            Arc::new(Mutex::new(HashMap::new()));
+        let pending_requests_clone = pending_requests.clone();
+
+        let timeout = Duration::from_secs(timeout_secs.unwrap_or(30));
+
+        // Spawn background reader task
+        tokio::spawn(async move {
+            while let Some(msg) = read.next().await {
+                match msg {
+                    Ok(Message::Text(text)) => {
+                        if let Ok(response_body) = serde_json::from_str::<Value>(&text) {
+                            if let Some(id) = response_body.get("id").and_then(|v| v.as_u64()) {
+                                let mut pending = pending_requests_clone.lock().await;
+                                if let Some(tx) = pending.remove(&id) {
+                                    let result = if let Some(error) = response_body.get("error") {
+                                        Err(anyhow::anyhow!("MCP server error: {}", error))
+                                    } else {
+                                        response_body.get("result").cloned().ok_or_else(|| {
+                                            anyhow::anyhow!("No result in MCP response")
+                                        })
+                                    };
+                                    let _ = tx.send(result);
+                                }
+                            }
+                        }
+                    }
+                    Ok(Message::Close(_)) => break,
+                    Err(_) => break,
+                    _ => continue,
+                }
+            }
+            // If we are here, the connection is closed or error occurred
+            let mut pending = pending_requests_clone.lock().await;
+            for (_, tx) in pending.drain() {
+                let _ = tx.send(Err(anyhow::anyhow!("WebSocket connection closed")));
+            }
+        });
 
         Ok(Self {
-            write: std::sync::Arc::new(tokio::sync::Mutex::new(write)),
-            read: std::sync::Arc::new(tokio::sync::Mutex::new(read)),
-            request_id: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
+            write: Arc::new(Mutex::new(write)),
+            pending_requests,
+            request_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
+            timeout,
         })
     }
 }
@@ -936,6 +976,12 @@
             "method": method,
             "params": params
         });
+
+        let (tx, rx) = oneshot::channel();
+        {
+            let mut pending = self.pending_requests.lock().await;
+            pending.insert(id, tx);
+        }
 
         // Send request
         let message = Message::Text(serde_json::to_string(&request_body)?.into());
@@ -948,51 +994,21 @@
                 .map_err(|e| anyhow::anyhow!("Failed to send WebSocket message: {}", e))?;
         }
 
-        // Read response
-        loop {
-            let mut read = self.read.lock().await;
-            let msg = read
-                .next()
-                .await
-                .ok_or_else(|| anyhow::anyhow!("WebSocket connection closed"))?
-                .map_err(|e| anyhow::anyhow!("WebSocket read error: {}", e))?;
-            drop(read);
-
-            match msg {
-                Message::Text(text) => {
-                    let response_body: Value = serde_json::from_str(&text)?;
-
-                    // Check if this is the response to our request
-                    if let Some(response_id) = response_body.get("id").and_then(|v| v.as_u64()) {
-                        if response_id == id {
-                            // Check for JSON-RPC errors
-                            if let Some(error) = response_body.get("error") {
-                                return Err(anyhow::anyhow!("MCP server error: {}", error));
-                            }
-
-                            return response_body
-                                .get("result")
-                                .cloned()
-                                .ok_or_else(|| anyhow::anyhow!("No result in MCP response"));
-                        }
-                    }
-                    // If it's not our response, continue reading
-                }
-                Message::Binary(_) => {
-                    // Handle binary messages if needed, for now skip
-                    continue;
-                }
-                Message::Close(_) => {
-                    return Err(anyhow::anyhow!("WebSocket connection closed by server"));
-                }
-                Message::Ping(_) | Message::Pong(_) => {
-                    // Handle ping/pong frames, continue reading
-                    continue;
-                }
-                Message::Frame(_) => {
-                    // Raw frames, continue reading
-                    continue;
-                }
+        // Wait for response with timeout
+        match tokio::time::timeout(self.timeout, rx).await {
+            Ok(Ok(result)) => result,
+            Ok(Err(_)) => {
+                let mut pending = self.pending_requests.lock().await;
+                pending.remove(&id);
+                Err(anyhow::anyhow!("Response channel closed"))
+            }
+            Err(_) => {
+                let mut pending = self.pending_requests.lock().await;
+                pending.remove(&id);
+                Err(anyhow::anyhow!(
+                    "Request timed out after {} seconds",
+                    self.timeout.as_secs()
+                ))
             }
         }
     }
```

## [After Fix Terminal Proof]
Verification of the fix via `cargo check -p mistralrs-mcp` (compilation success) and logical verification that multiplexing via a background reader task and `oneshot` channels correctly correlates responses and prevents race conditions.

```bash
$ cargo check -p mistralrs-mcp
    Checking mistralrs-mcp v0.7.1-alpha.1 (/app/mistralrs-mcp)
    Finished `dev` profile [optimized + debuginfo] target(s) in 21.47s
```
