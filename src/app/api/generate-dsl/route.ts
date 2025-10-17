import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { gameName, gameDescription } = body;

    if (!gameName || typeof gameName !== "string" || !gameDescription || typeof gameDescription !== "string") {
      return NextResponse.json(
        { success: false, error: "Game name and description are required" },
        { status: 400 }
      );
    }

    // 统一文件名生成逻辑
    const filename = `${gameName.trim().toLowerCase().replace(/\s+/g, '-')}.yaml`;

    // Step 1: Create a thread
    const threadResponse = await fetch("http://localhost:8123/threads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    if (!threadResponse.ok) {
      throw new Error(`Failed to create thread: ${threadResponse.status}`);
    }

    const { thread_id: threadId } = await threadResponse.json();

    // Step 2: Create a run
    const runResponse = await fetch(`http://localhost:8123/threads/${threadId}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        assistant_id: "dsl_agent",
        input: {
          game_description: gameDescription.trim(),
          yaml_path: filename
        },
      }),
    });

    if (!runResponse.ok) {
      throw new Error(`DSL agent error: ${runResponse.status}`);
    }

    const { run_id: runId } = await runResponse.json();

    // Step 3: Poll for completion
    const maxAttempts = 900; // 15 minutes for GPT-5 processing
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const statusResponse = await fetch(`http://localhost:8123/threads/${threadId}/runs/${runId}`);
      
      if (!statusResponse.ok) {
        attempts++;
        continue;
      }

      const runStatus = await statusResponse.json();
      console.log(`Run status attempt ${attempts + 1}: ${runStatus.status}`);
      
      if (runStatus.status === "success") {
        // Step 4: Read generated file
        const fs = await import('fs').then(m => m.promises);
        const path = await import('path');
        const filePath = path.join(process.cwd(), 'games', filename);
        
        const dslYaml = await fs.readFile(filePath, 'utf-8');
        
        return NextResponse.json({
          success: true,
          dsl: dslYaml,
          filename: filename,
          message: `DSL for "${gameName}" has been successfully generated`,
        });
      } else if (runStatus.status === "error" || runStatus.status === "failed") {
        throw new Error(`DSL generation failed: ${runStatus.error || "Unknown error"}`);
      }
      
      attempts++;
    }

    throw new Error("DSL generation timeout after 15 minutes");

  } catch (error) {
    console.error("DSL generation error:", error);
    
    return NextResponse.json(
      { 
        success: false, 
        error: "Failed to generate DSL. Please check if LangGraph server is running on port 8123." 
      },
      { status: 500 }
    );
  }
}