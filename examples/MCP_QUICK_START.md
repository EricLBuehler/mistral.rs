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
        "args": ["@modelcontextprotocol/server-filesystem", "/tmp"]
      }
    }
  ],
  "auto_register_tools": true
}
```

**Alternative: Web Search Example**
```json
{
  "servers": [
    {
      "name": "Brave Search",
      "source": {
        "type": "Process", 
        "command": "npx",
        "args": ["@modelcontextprotocol/server-brave-search"]
      },
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key"
      }
    }
  ],
  "auto_register_tools": true
}
```

### 2. Start Server with MCP Tools

```bash
# Start with MCP configuration
mistralrs-server --mcp-config mcp-config.json --port 1234 run -m Qwen/Qwen3-4B

# Alternative: Use environment variable
export MCP_CONFIG_PATH=mcp-config.json
mistralrs-server --port 1234 run -m Qwen/Qwen3-4B
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
      {"role": "user", "content": "List the files in /tmp and create a hello.txt file"}
    ]
  }'
```

```bash  
# Web search example (if using Brave search config)
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B", 
    "messages": [
      {"role": "user", "content": "Search for the latest AI news"}
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

## 🚀 Popular MCP Servers

| Server | Description | Installation | Use Case |
|--------|-------------|--------------|----------|
| **Filesystem** | File operations | `npm i -g @modelcontextprotocol/server-filesystem` | Read/write files |
| **Brave Search** | Web search | `npm i -g @modelcontextprotocol/server-brave-search` | Search the web |
| **GitHub** | GitHub API | `npm i -g @modelcontextprotocol/server-github` | Repository access |
| **Postgres** | Database | `npm i -g @modelcontextprotocol/server-postgres` | SQL queries |

### Transport Types
| Transport | When to Use | Examples |
|-----------|-------------|----------|
| **Process** | Local tools, npm packages | Most MCP servers |
| **HTTP** | REST APIs, cloud services | Custom web services |
| **WebSocket** | Real-time streaming | Live data feeds |

## 📋 Ready-to-Use Configurations

### Filesystem Server (Recommended for testing)
```json
{
  "servers": [{
    "name": "File Operations",
    "source": {
      "type": "Process",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }],
  "auto_register_tools": true
}
```

### Web Search Server
```json
{
  "servers": [{
    "name": "Brave Search",
    "source": {
      "type": "Process",
      "command": "npx", 
      "args": ["@modelcontextprotocol/server-brave-search"]
    },
    "env": {
      "BRAVE_API_KEY": "your-api-key-here"
    }
  }],
  "auto_register_tools": true
}
```

### Multi-Server Setup
```json
{
  "servers": [
    {
      "name": "Filesystem",
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-filesystem", "/workspace"]
      },
      "tool_prefix": "fs"
    },
    {
      "name": "GitHub", 
      "source": {
        "type": "Process",
        "command": "npx",
        "args": ["@modelcontextprotocol/server-github"]
      },
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token"
      },
      "tool_prefix": "gh"
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