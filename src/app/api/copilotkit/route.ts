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
  // Get room-scoped threadId from request header
  const headerThreadId = req.headers.get('X-Thread-ID');
  if (!headerThreadId) {
    console.warn('⚠️ Missing X-Thread-ID header, falling back to "default". This may cause shared chat history across rooms.');
  }
  const threadId = headerThreadId || 'default';
  console.log('🧵 Using threadId for this request:', threadId);
  
  // Create a runtime bound to the specific threadId per request
  const dynamicRuntime = new CopilotRuntime({
    agents: {
      "sample_agent": new LangGraphAgent({
        ...baseConfig,        // Reuse base config
        threadId: threadId,   // 🔑 Room-scoped thread ID
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