use mistralrs_mcp::transport::{HttpTransport, McpTransport};
use serde_json::{json, Value};
use std::{collections::HashMap, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

#[tokio::test]
async fn http_identity_preserves_custom_headers() -> anyhow::Result<()> {
    for custom_identity in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let url = format!("http://{}", listener.local_addr()?);
        let expected = if custom_identity {
            "custom-client/1".to_string()
        } else {
            format!("mistralrs/{}", env!("CARGO_PKG_VERSION"))
        };
        let capture = tokio::spawn(async move {
            for method in ["tools/call", "tools/call", "notifications/initialized"] {
                let (mut socket, _) = listener.accept().await?;
                let mut bytes = Vec::new();
                loop {
                    bytes.push(socket.read_u8().await?);
                    if bytes.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let headers = String::from_utf8(bytes)?.to_ascii_lowercase();
                assert!(headers.contains(&format!("user-agent: {expected}\r\n")));
                assert!(headers.contains("authorization: bearer test-token\r\n"));
                assert!(headers.contains("x-test: kept\r\n"));
                let len: usize = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .unwrap()
                    .parse()?;
                let mut body = vec![0; len];
                socket.read_exact(&mut body).await?;
                let body: Value = serde_json::from_slice(&body)?;
                assert_eq!(body["method"], method);
                let response = json!({"jsonrpc":"2.0","id":body["id"],"result":{}}).to_string();
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", response.len(), response).as_bytes()).await?;
            }
            Ok::<_, anyhow::Error>(())
        });
        let mut headers = HashMap::from([
            ("Authorization".to_string(), "Bearer test-token".to_string()),
            ("X-Test".to_string(), "kept".to_string()),
        ]);
        if custom_identity {
            headers.insert("User-Agent".to_string(), "custom-client/1".to_string());
        }
        let transport = HttpTransport::new(url, Some(5), Some(headers))?;
        for name in ["web_search", "web_fetch"] {
            transport
                .send_request("tools/call", json!({"name": name, "arguments":{}}))
                .await?;
        }
        transport.send_initialization_notification().await?;
        tokio::time::timeout(Duration::from_secs(5), capture).await???;
    }
    Ok(())
}
