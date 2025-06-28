#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListResourcesRequestSchema, ListToolsRequestSchema, ReadResourceRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
// Define the tools once to avoid repetition
const TOOLS = [
    
    {
        name: "math_operation",
        description: "Perform a math operation",
        inputSchema: {
            type: "object",
            properties: {
                operation: { type: "string", description: "operation like add, subtract, multiplier and devider" },
                num1: { type: "number", description: "num1 which is number. it is used to perform operation like num1 add num2" },
                num2: { type: "number", description: "num2 which is number. it is second operand used to perform operation like num1 add num2 num1 + num2" },
            },
            required: ["operation", "num1", "num2"],
        },
    },
    {
        name: "task_finder",
        description: "Find software story based on given imported keyword query from vector database through embedding.",
        inputSchema: {
            type: "object",
            properties: {
                query: { type: "string", description: "Search Query based on which it find macching tasks from vector db." },
            },
            required: ["query"],
        },
    }
];
const consoleLogs = [];
// Deep merge utility function

async function handleToolCall(name, args) {
    debugger;
    console.error("Handle tool call CJ : ", name, args);

    switch (name) {
        case "math_operation": {
            const { operation, num1, num2 } = args;
            let result;
            switch (operation) {
                case "add":
                    result = num1 + num2;
                    break;
                case "subtract":
                    result = num1 - num2;
                    break;
                case "multiply":
                    result = num1 * num2;
                    break;
                case "divide":
                    if (operand2 === 0) {
                        return {
                            content: [{
                                    type: "text",
                                    text: "Division by zero is not allowed.",
                                }],
                            isError: true,
                        };
                    }
                    result = num1 / num2;
                    break;
                default:
                    return {
                        content: [{
                                type: "text",
                                text: `Unknown operation: ${operation}`,
                            }],
                        isError: true,
                    };
            }
            return {
                content: [{
                        type: "text",
                        text: `Result of ${operation} ${num1} and ${num2} is ${result}`,
                    }],
                isError: false,
            };
        }
        case "task_finder": {
            const { query } = args;
            const response = await fetch(`http://127.0.0.1:5000/api?query=${encodeURIComponent(query)}`);
            const data = await response.text(); // or response.text()
            return {
                content: [{
                        type: "text",
                        text: data
                    }],
                isError: false,
            };
        }
        default:
            return {
                content: [{
                        type: "text",
                        text: `Unknown tool: ${name}`,
                    }],
                isError: true,
            };
    }
}
const server = new Server({
    name: "example-servers/puppeteer",
    version: "0.1.0",
}, {
    capabilities: {
        resources: {},
        tools: {},
    },
});

// Setup request handlers
server.setRequestHandler(ListResourcesRequestSchema, async () => ({
    resources: [
        {
            uri: "console://logs",
            mimeType: "text/plain",
            name: "Browser console logs",
        },
        ...Array.from(screenshots.keys()).map(name => ({
            uri: `screenshot://${name}`,
            mimeType: "image/png",
            name: `Screenshot: ${name}`,
        })),
    ],
}));
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
    const uri = request.params.uri.toString();
    if (uri === "console://logs") {
        return {
            contents: [{
                    uri,
                    mimeType: "text/plain",
                    text: consoleLogs.join("\n"),
                }],
        };
    }
    throw new Error(`Resource not found: ${uri}`);
});
server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS,
}));
server.setRequestHandler(CallToolRequestSchema, async (request) => handleToolCall(request.params.name, request.params.arguments ?? {}));
async function runServer() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
}
runServer().catch(console.error);
process.stdin.on("close", () => {
    console.error("Puppeteer MCP Server closed");
    server.close();
});