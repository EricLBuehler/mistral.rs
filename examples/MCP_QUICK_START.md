# MCP Quick Start Guide

🔗 **Connect your models to external tools and services** using the Model Context Protocol (MCP).

**What is MCP?** The Model Context Protocol allows your AI models to access external tools like web search, file systems, databases, and APIs automatically during conversations.

**Key Benefits:**
- 🚀 **Zero setup** - Tools work automatically once configured
- 🔧 **Multi-tool support** - Connect to multiple services simultaneously  
- 🌐 **Universal protocol** - Works with any MCP-compatible server
- 🔒 **Secure** - Built-in authentication and timeout controls

[📚 Full Documentation](../docs/MCP/README.md) | [⚙️ Configuration Reference](../docs/MCP/configuration.md)

## Quick Start

### 1. Create Configuration File

Create `mcp-config.json` with a **working example** using the filesystem server:

```json
{
  "servers": [
    {
      "name": "Filesystem Tools",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ],
  "auto_register_tools": true
}
```

> **Note:** To install the filesystem server, run: `npx @modelcontextprotocol/server-filesystem . -y`

**Alternative Transport Examples (commented out by default):**

<details>
<summary>HTTP Example - Hugging Face MCP Server</summary>

```json
{
  "servers": [
    {
      "name": "Hugging Face MCP",
      "source": {
        "type": "Http",
        "url": "https://hf.co/mcp",
        "timeout_secs": 30
      },
      "bearer_token": "hf_xxx",
      "tool_prefix": "hf",
      "enabled": false
    },
    {
      "name": "Filesystem Tools",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ],
  "auto_register_tools": true
}
```
</details>

<details>
<summary>WebSocket Example</summary>

```json
{
  "servers": [
    {
      "name": "WebSocket Example",
      "source": {
        "type": "WebSocket",
        "url": "wss://api.example.com/mcp",
        "timeout_secs": 30
      },
      "enabled": false
    },
    {
      "name": "Filesystem Tools",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ],
  "auto_register_tools": true
}
```
</details>

### 2. Start Server with MCP Tools

```bash
# Start with MCP configuration
mistralrs serve --mcp-config mcp-config.json -p 1234 -m Qwen/Qwen3-4B

# Alternative: Use environment variable
export MCP_CONFIG_PATH=mcp-config.json
mistralrs serve -p 1234 -m Qwen/Qwen3-4B
```

✅ **Server starts with tools automatically loaded!**

### 3. Use Tools Automatically

**That's it! Tools work automatically in conversations:**

```bash
# File operations example
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [
      {"role": "user", "content": "List the files in the current directory and create a hello.txt file"}
    ]
  }'
```

🎉 **The model automatically uses tools when needed - no extra steps required!**

## ✅ Verify Tools are Working

```bash
curl http://localhost:1234/v1/models
```

Look for MCP status in the response:
```json
{
  "object": "list",
  "data": [{
    "id": "Qwen/Qwen3-4B",
    "object": "model", 
    "tools_available": true,
    "mcp_tools_count": 3,
    "mcp_servers_connected": 1
  }]
}
```

**✅ `tools_available: true` means MCP tools are working!**

## 🔧 Quick Verification

**Test filesystem server is working:**
```bash
# This should return "3" or more (filesystem tools available)
curl http://localhost:1234/v1/models | jq '.data[0].mcp_tools_count'

# Check if filesystem server process is running
ps aux | grep server-filesystem
```

## 🚀 Popular MCP Servers

| Server | Description | Installation | Use Case |
|--------|-------------|--------------|----------|
| **Filesystem** | File operations | `npm i -g @modelcontextprotocol/server-filesystem` | Read/write files |
| **Hugging Face** | HF API access | Web service at `https://hf.co/mcp` | Models, datasets, spaces |
| **Postgres** | Database | `npm i -g @modelcontextprotocol/server-postgres` | SQL queries |

> **Links to more servers:**
> - [Brave Search](https://github.com/modelcontextprotocol/servers/tree/main/src/brave-search) - Web search capabilities
> - [GitHub](https://github.com/modelcontextprotocol/servers/tree/main/src/github) - Repository access

### Transport Types
| Transport | When to Use | Examples |
|-----------|-------------|----------|
| **Process** | Local tools, npm packages | Most MCP servers |
| **HTTP** | REST APIs, cloud services | Custom web services |
| **WebSocket** | Real-time streaming | Live data feeds |

## 📋 Ready-to-Use Configurations

### Process Example (Default - Filesystem Server)
```json
{
  "servers": [{
    "name": "File Operations",
    "source": {
      "type": "Process",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "."]
    }
  }],
  "auto_register_tools": true
}
```

### HTTP Example (Hugging Face MCP Server)
```json
{
  "servers": [
    {
      "name": "Hugging Face MCP",
      "source": {
        "type": "Http",
        "url": "https://hf.co/mcp",
        "timeout_secs": 30
      },
      "bearer_token": "hf_xxx",
      "tool_prefix": "hf",
      "enabled": false
    },
    {
      "name": "File Operations",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ],
  "auto_register_tools": true
}
```

### WebSocket Example
```json
{
  "servers": [
    {
      "name": "WebSocket Example",
      "source": {
        "type": "WebSocket",
        "url": "wss://api.example.com/mcp",
        "timeout_secs": 30
      },
      "enabled": false
    },
    {
      "name": "File Operations",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "."]
      }
    }
  ],
  "auto_register_tools": true
}
```

## Error Handling

The system gracefully handles failures:
- **Startup**: Invalid configurations are caught and reported with helpful messages
- **Runtime**: Failed MCP connections are logged as warnings, server continues without MCP
- **Tools**: Individual tool failures don't crash the server

## 📚 Next Steps

**Ready for more?**
- 🔧 [Configuration Reference](mcp-config-reference.json) - All available options
- 📖 [Full MCP Documentation](../docs/MCP/README.md) - Complete guide  
- 🛠️ [Server Examples](mcp-server-config.json) - Real-world configurations
- 🚀 [Advanced Usage](../docs/MCP/advanced.md) - Multi-server setups

**Need help?** 
- [MCP Server Registry](https://github.com/modelcontextprotocol/servers) - Find more servers
- [Troubleshooting](../docs/MCP/README.md#troubleshooting) - Common issues
- [Discord Community](https://discord.gg/SZrecqK8qw) - Get support