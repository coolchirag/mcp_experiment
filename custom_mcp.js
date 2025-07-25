#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListResourcesRequestSchema, ListToolsRequestSchema, ReadResourceRequestSchema, } from "@modelcontextprotocol/sdk/types.js";
import puppeteer from "puppeteer";
// Define the tools once to avoid repetition


const TOOLS = [
    {
        name: "windows_os_command_executor",
        description: "Execute windows OS command",
        inputSchema: {
            type: "object",
            properties: {
                command: { type: "string", description: "SIngle line Windows CMD command" }
                
            },
            required: ["command"],
        },
    },
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
        name: "puppeteer_navigate",
        description: "Navigate to a URL",
        inputSchema: {
            type: "object",
            properties: {
                url: { type: "string", description: "URL to navigate to" },
                launchOptions: { type: "object", description: "PuppeteerJS LaunchOptions. Default null. If changed and not null, browser restarts. Example: { headless: true, args: ['--no-sandbox'] }" },
                allowDangerous: { type: "boolean", description: "Allow dangerous LaunchOptions that reduce security. When false, dangerous args like --no-sandbox will throw errors. Default false." },
            },
            required: ["url"],
        },
    },
    {
        name: "puppeteer_screenshot",
        description: "Take a screenshot of the current page or a specific element",
        inputSchema: {
            type: "object",
            properties: {
                name: { type: "string", description: "Name for the screenshot" },
                selector: { type: "string", description: "CSS selector for element to screenshot" },
                width: { type: "number", description: "Width in pixels (default: 800)" },
                height: { type: "number", description: "Height in pixels (default: 600)" },
                encoded: { type: "boolean", description: "If true, capture the screenshot as a base64-encoded data URI (as text) instead of binary image content. Default false." },
            },
            required: ["name"],
        },
    },
    {
        name: "puppeteer_click",
        description: "Click an element on the page",
        inputSchema: {
            type: "object",
            properties: {
                selector: { type: "string", description: "CSS selector for element to click" },
            },
            required: ["selector"],
        },
    },
    {
        name: "puppeteer_fill",
        description: "Fill out an input field",
        inputSchema: {
            type: "object",
            properties: {
                selector: { type: "string", description: "CSS selector for input field" },
                value: { type: "string", description: "Value to fill" },
            },
            required: ["selector", "value"],
        },
    },
    {
        name: "puppeteer_select",
        description: "Select an element on the page with Select tag",
        inputSchema: {
            type: "object",
            properties: {
                selector: { type: "string", description: "CSS selector for element to select" },
                value: { type: "string", description: "Value to select" },
            },
            required: ["selector", "value"],
        },
    },
    {
        name: "puppeteer_hover",
        description: "Hover an element on the page",
        inputSchema: {
            type: "object",
            properties: {
                selector: { type: "string", description: "CSS selector for element to hover" },
            },
            required: ["selector"],
        },
    },
    {
        name: "puppeteer_evaluate",
        description: "Execute JavaScript in the browser console",
        inputSchema: {
            type: "object",
            properties: {
                script: { type: "string", description: "JavaScript code to execute" },
            },
            required: ["script"],
        },
    },
];
// Global state
let browser;
let page;
const consoleLogs = [];
const screenshots = new Map();
let previousLaunchOptions = null;
async function ensureBrowser({ launchOptions, allowDangerous }) {
    const DANGEROUS_ARGS = [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--single-process',
        '--disable-web-security',
        '--ignore-certificate-errors',
        '--disable-features=IsolateOrigins',
        '--disable-site-isolation-trials',
        '--allow-running-insecure-content'
    ];
    // Parse environment config safely
    let envConfig = {};
    try {
        envConfig = JSON.parse(process.env.PUPPETEER_LAUNCH_OPTIONS || '{}');
    }
    catch (error) {
        console.warn('Failed to parse PUPPETEER_LAUNCH_OPTIONS:', error?.message || error);
    }
    // Deep merge environment config with user-provided options
    const mergedConfig = deepMerge(envConfig, launchOptions || {});
    // Security validation for merged config
    if (mergedConfig?.args) {
        const dangerousArgs = mergedConfig.args?.filter?.((arg) => DANGEROUS_ARGS.some((dangerousArg) => arg.startsWith(dangerousArg)));
        if (dangerousArgs?.length > 0 && !(allowDangerous || (process.env.ALLOW_DANGEROUS === 'true'))) {
            throw new Error(`Dangerous browser arguments detected: ${dangerousArgs.join(', ')}. Fround from environment variable and tool call argument. ` +
                'Set allowDangerous: true in the tool call arguments to override.');
        }
    }
    try {
        if ((browser && !browser.connected) ||
            (launchOptions && (JSON.stringify(launchOptions) != JSON.stringify(previousLaunchOptions)))) {
            await browser?.close();
            browser = null;
        }
    }
    catch (error) {
        browser = null;
    }
    previousLaunchOptions = launchOptions;
    if (!browser) {
        const npx_args = { headless: false };
        const docker_args = { headless: true, args: ["--no-sandbox", "--single-process", "--no-zygote"] };
        browser = await puppeteer.launch(deepMerge(process.env.DOCKER_CONTAINER ? docker_args : npx_args, mergedConfig));
        const pages = await browser.pages();
        page = pages[0];
        page.on("console", (msg) => {
            const logEntry = `[${msg.type()}] ${msg.text()}`;
            consoleLogs.push(logEntry);
            server.notification({
                method: "notifications/resources/updated",
                params: { uri: "console://logs" },
            });
        });
    }
    return page;
}
// Deep merge utility function
function deepMerge(target, source) {
    const output = Object.assign({}, target);
    if (typeof target !== 'object' || typeof source !== 'object')
        return source;
    for (const key of Object.keys(source)) {
        const targetVal = target[key];
        const sourceVal = source[key];
        if (Array.isArray(targetVal) && Array.isArray(sourceVal)) {
            // Deduplicate args/ignoreDefaultArgs, prefer source values
            output[key] = [...new Set([
                    ...(key === 'args' || key === 'ignoreDefaultArgs' ?
                        targetVal.filter((arg) => !sourceVal.some((launchArg) => arg.startsWith('--') && launchArg.startsWith(arg.split('=')[0]))) :
                        targetVal),
                    ...sourceVal
                ])];
        }
        else if (sourceVal instanceof Object && key in target) {
            output[key] = deepMerge(targetVal, sourceVal);
        }
        else {
            output[key] = sourceVal;
        }
    }
    return output;
}
async function handleToolCall(name, args) {
    debugger;
    console.error("Handle tool call CJ : ", name, args);

    let page = null;
    if (name !== "math_operation" && name !== "windows_os_command_executor" ) {
       page = await ensureBrowser(args);
    }
    switch (name) {
        case "windows_os_command_executor":{
            let isError = false;
            let response_msg = '';
            try {
                    const response = await fetch('http://localhost:3000/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ "command": args.command }),
                    });

                    // Check if the response is OK (status 200)
                    if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                    }

                    const data = await response.json();

                    // Handle the API response
                    if (data.error) {
                    console.error('Error:', data.error);
                    if (data.stderr) {
                        console.error('Stderr:', data.stderr);
                    }
                    
                    }
                    response_msg = data.stdout;
                    console.log('Output:', data.stdout);
                } catch (error) {
                    isError = true;
                    response_msg='Request failed : '+error.message;
                    console.error('Request failed:', error.message);
                }
                return {
                        content: [{
                                type: "text",
                                text: `Command execution resault : ${response_msg}`,
                            }],
                        isError: isError,
                        };
        }
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
        case "puppeteer_navigate":
            await page.goto(args.url);
            return {
                content: [{
                        type: "text",
                        text: `Navigated to ${args.url}`,
                    }],
                isError: false,
            };
        case "puppeteer_screenshot": {
            const width = args.width ?? 800;
            const height = args.height ?? 600;
            const encoded = args.encoded ?? false;
            await page.setViewport({ width, height });
            const screenshot = await (args.selector ?
                (await page.$(args.selector))?.screenshot({ encoding: "base64" }) :
                page.screenshot({ encoding: "base64", fullPage: false }));
            if (!screenshot) {
                return {
                    content: [{
                            type: "text",
                            text: args.selector ? `Element not found: ${args.selector}` : "Screenshot failed",
                        }],
                    isError: true,
                };
            }
            screenshots.set(args.name, screenshot);
            server.notification({
                method: "notifications/resources/list_changed",
            });
            return {
                content: [
                    {
                        type: "text",
                        text: `Screenshot '${args.name}' taken at ${width}x${height}`,
                    },
                    encoded ? {
                        type: "text",
                        text: `data:image/png;base64,${screenshot}`,
                    } : {
                        type: "image",
                        data: screenshot,
                        mimeType: "image/png",
                    },
                ],
                isError: false,
            };
        }
        case "puppeteer_click":
            try {
                await page.click(args.selector);
                return {
                    content: [{
                            type: "text",
                            text: `Clicked: ${args.selector}`,
                        }],
                    isError: false,
                };
            }
            catch (error) {
                return {
                    content: [{
                            type: "text",
                            text: `Failed to click ${args.selector}: ${error.message}`,
                        }],
                    isError: true,
                };
            }
        case "puppeteer_fill":
            try {
                await page.waitForSelector(args.selector);
                await page.type(args.selector, args.value);
                return {
                    content: [{
                            type: "text",
                            text: `Filled ${args.selector} with: ${args.value}`,
                        }],
                    isError: false,
                };
            }
            catch (error) {
                return {
                    content: [{
                            type: "text",
                            text: `Failed to fill ${args.selector}: ${error.message}`,
                        }],
                    isError: true,
                };
            }
        case "puppeteer_select":
            try {
                await page.waitForSelector(args.selector);
                await page.select(args.selector, args.value);
                return {
                    content: [{
                            type: "text",
                            text: `Selected ${args.selector} with: ${args.value}`,
                        }],
                    isError: false,
                };
            }
            catch (error) {
                return {
                    content: [{
                            type: "text",
                            text: `Failed to select ${args.selector}: ${error.message}`,
                        }],
                    isError: true,
                };
            }
        case "puppeteer_hover":
            try {
                await page.waitForSelector(args.selector);
                await page.hover(args.selector);
                return {
                    content: [{
                            type: "text",
                            text: `Hovered ${args.selector}`,
                        }],
                    isError: false,
                };
            }
            catch (error) {
                return {
                    content: [{
                            type: "text",
                            text: `Failed to hover ${args.selector}: ${error.message}`,
                        }],
                    isError: true,
                };
            }
        case "puppeteer_evaluate":
            try {
                await page.evaluate(() => {
                    window.mcpHelper = {
                        logs: [],
                        originalConsole: { ...console },
                    };
                    ['log', 'info', 'warn', 'error'].forEach(method => {
                        console[method] = (...args) => {
                            window.mcpHelper.logs.push(`[${method}] ${args.join(' ')}`);
                            window.mcpHelper.originalConsole[method](...args);
                        };
                    });
                });
                const result = await page.evaluate(args.script);
                const logs = await page.evaluate(() => {
                    Object.assign(console, window.mcpHelper.originalConsole);
                    const logs = window.mcpHelper.logs;
                    delete window.mcpHelper;
                    return logs;
                });
                return {
                    content: [
                        {
                            type: "text",
                            text: `Execution result:\n${JSON.stringify(result, null, 2)}\n\nConsole output:\n${logs.join('\n')}`,
                        },
                    ],
                    isError: false,
                };
            }
            catch (error) {
                return {
                    content: [{
                            type: "text",
                            text: `Script execution failed: ${error.message}`,
                        }],
                    isError: true,
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
    if (uri.startsWith("screenshot://")) {
        const name = uri.split("://")[1];
        const screenshot = screenshots.get(name);
        if (screenshot) {
            return {
                contents: [{
                        uri,
                        mimeType: "image/png",
                        blob: screenshot,
                    }],
            };
        }
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

function f1() {
    runServer();
}
