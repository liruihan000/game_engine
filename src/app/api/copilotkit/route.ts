import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";

import { LangGraphAgent } from "@copilotkit/runtime";
import { NextRequest } from "next/server";
 
// 1. You can use any service adapter here for multi-agent support. We use
//    the empty adapter since we're only using one agent.
const serviceAdapter = new ExperimentalEmptyAdapter();

// 2. Base configuration for LangGraph Agent
const baseConfig = {
  deploymentUrl: process.env.LANGGRAPH_DEPLOYMENT_URL || "http://localhost:8123",
  graphId: "sample_agent",
  langsmithApiKey: process.env.LANGSMITH_API_KEY || "",
};
 
// 3. Build a Next.js API route that handles the CopilotKit runtime requests.
export const POST = async (req: NextRequest) => {
  // 从请求头获取房间特定的 threadId
  const threadId = req.headers.get('X-Thread-ID') || 'default';
  console.log('🧵 Using threadId for this request:', threadId);
  
  // 为每个请求创建带有特定 threadId 的 runtime
  const dynamicRuntime = new CopilotRuntime({
    agents: {
      "sample_agent": new LangGraphAgent({
        ...baseConfig,        // 复用基础配置
        threadId: threadId,   // 🔑 使用房间特定的线程ID
      }),
    }
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime: dynamicRuntime, 
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });
 
  return handleRequest(req);
};