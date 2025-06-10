#!/bin/bash

# Example demonstrating how to use mistral.rs HTTP server with MCP client support
# 
# This example shows how to:
# 1. Start the mistral.rs server with MCP configuration
# 2. Send chat requests via HTTP that can automatically use MCP tools
# 3. Parse responses to see tool calls made to MCP servers

echo "MCP Client HTTP API Example"
echo "==========================="
echo ""

# Check if server is running
if ! curl -s http://localhost:1234/health > /dev/null 2>&1; then
    echo "❌ mistral.rs server is not running on port 1234"
    echo ""
    echo "Please start the server first with MCP configuration:"
    echo "cargo run --release --bin mistralrs-server -- \\"
    echo "  --port 1234 \\"
    echo "  plain \\"
    echo "  -m Qwen/Qwen3-4B \\"
    echo "  -a qwen3 \\"
    echo "  --mcp-config examples/mcp-test-config.json"
    echo ""
    echo "Make sure examples/mcp-test-config.json exists and contains:"
    echo "{"
    echo "  \"servers\": ["
    echo "    {"
    echo "      \"id\": \"hf_server\","
    echo "      \"name\": \"Hugging Face MCP Server\","
    echo "      \"source\": {"
    echo "        \"type\": \"Http\","
    echo "        \"url\": \"https://hf.co/mcp\","
    echo "        \"timeout_secs\": 30"
    echo "      },"
    echo "      \"enabled\": true,"
    echo "      \"tool_prefix\": \"hf\","
    echo "      \"bearer_token\": \"hf_xxx\""
    echo "    }"
    echo "  ],"
    echo "  \"auto_register_tools\": true,"
    echo "  \"tool_timeout_secs\": 30,"
    echo "  \"max_concurrent_calls\": 5"
    echo "}"
    exit 1
fi

echo "✅ mistral.rs server is running on port 1234"
echo ""

# Check if MCP tools are available
echo "🔍 Checking available models and MCP status..."
MODEL_INFO=$(curl -s http://localhost:1234/v1/models)
echo "Models endpoint response: $MODEL_INFO"
echo ""

echo "📤 Sending chat request with MCP tool support..."
echo "The model will automatically use MCP tools if needed to answer the question."
echo ""

# Send chat completion request that should trigger MCP tool usage
RESPONSE=$(curl -s -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer placeholder" \
  -d '{
    "model": "mistral",
    "messages": [
      {
        "role": "system",
        "content": "You are an AI assistant with access to external tools via MCP servers. You can access Hugging Face tools and other external services. Use these tools when appropriate to help answer user questions."
      },
      {
        "role": "user",
        "content": "Hello! Can you help me get the top 10 HF models right now?"
      }
    ],
    "max_tokens": 1000,
    "temperature": 0.1,
    "tool_choice": "auto"
  }')

echo "📥 Response received:"
echo "==================="

# Pretty print the response
if command -v jq &> /dev/null; then
    echo "$RESPONSE" | jq '.'
    
    # Extract and display tool calls if any
    TOOL_CALLS=$(echo "$RESPONSE" | jq -r '.choices[0].message.tool_calls // empty')
    if [ -n "$TOOL_CALLS" ] && [ "$TOOL_CALLS" != "null" ]; then
        echo ""
        echo "🔧 MCP tool calls made:"
        echo "-----------------------"
        echo "$RESPONSE" | jq -r '.choices[0].message.tool_calls[] | "Tool: \(.function.name)\nArguments: \(.function.arguments)\n"'
    else
        echo ""
        echo "ℹ️  No tool calls were made for this request."
        echo "   (This could mean MCP servers aren't configured or the question didn't require tool usage)"
    fi
    
    # Display usage information
    USAGE=$(echo "$RESPONSE" | jq -r '.usage // empty')
    if [ -n "$USAGE" ] && [ "$USAGE" != "null" ]; then
        echo ""
        echo "📊 Token usage:"
        echo "$RESPONSE" | jq -r '.usage | "Total tokens: \(.total_tokens // "unknown")\nCompletion tokens: \(.completion_tokens // "unknown")\nPrompt tokens: \(.prompt_tokens // "unknown")"'
    fi
else
    echo "$RESPONSE"
    echo ""
    echo "💡 Tip: Install 'jq' for better JSON formatting: sudo apt-get install jq"
fi

echo ""
echo "✅ Example completed!"
echo ""
echo "For more information about MCP configuration, see:"
echo "- examples/mcp-test-config.json (simple config)"
echo "- examples/mcp-server-config.json (advanced config with multiple transports)"
echo "- mistralrs/examples/mcp_client/main.rs (Rust API usage)"
echo "- examples/python/mcp_client_example.py (Python API usage)"