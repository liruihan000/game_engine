"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Download, FileText } from "lucide-react";

export default function DSLGeneratorPage() {
  const [gameName, setGameName] = useState("");
  const [gameDescription, setGameDescription] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedDSL, setGeneratedDSL] = useState("");
  const [filename, setFilename] = useState("generated-game.yaml");
  const [error, setError] = useState("");

  const handleGenerate = async () => {
    if (!gameName.trim() || !gameDescription.trim()) {
      setError("Please enter both game name and description");
      return;
    }

    setIsGenerating(true);
    setError("");
    setGeneratedDSL("");

    try {
      // Call DSL generation API
      const response = await fetch("/api/generate-dsl", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          gameName: gameName.trim(),
          gameDescription: gameDescription.trim() 
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        setGeneratedDSL(result.dsl);
        setFilename(result.filename || `${gameName.trim().toLowerCase().replace(/\s+/g, '-')}.yaml`);
      } else {
        setError(result.error || "Failed to generate DSL");
      }
    } catch (err) {
      console.error("DSL generation error:", err);
      setError("Failed to generate DSL. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!generatedDSL) return;

    const blob = new Blob([generatedDSL], { type: "application/x-yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold">DSL Generator</h1>
          <p className="text-muted-foreground">
            Describe your game concept and generate a structured DSL file
          </p>
        </div>

        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle>Game Description</CardTitle>
            <CardDescription>
              Describe your game mechanics, rules, and player interactions. 
              Be specific about phases, win conditions, and player actions.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label htmlFor="gameName" className="text-sm font-medium block mb-2">
                Game Name
              </label>
              <Input
                id="gameName"
                placeholder="e.g., Werewolf, Mafia, Among Us"
                value={gameName}
                onChange={(e) => setGameName(e.target.value)}
                disabled={isGenerating}
              />
            </div>
            
            <div>
              <label htmlFor="gameDescription" className="text-sm font-medium block mb-2">
                Game Description
              </label>
              <Textarea
                id="gameDescription"
                placeholder="Example: This is a multiplayer social deduction game where players are assigned roles like werewolf, seer, and villager. During the night phase, werewolves eliminate a player. During the day phase, all players vote to eliminate someone they suspect is a werewolf..."
                value={gameDescription}
                onChange={(e) => setGameDescription(e.target.value)}
                className="min-h-32"
                disabled={isGenerating}
              />
            </div>
            
            {error && (
              <div className="text-sm text-red-600 bg-red-50 p-3 rounded-md border border-red-200">
                {error}
              </div>
            )}

            <Button 
              onClick={handleGenerate}
              disabled={isGenerating || !gameName.trim() || !gameDescription.trim()}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating DSL...
                </>
              ) : (
                <>
                  <FileText className="mr-2 h-4 w-4" />
                  Generate DSL
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Output Section */}
        {generatedDSL && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Generated DSL</CardTitle>
                  <CardDescription>
                    Your game DSL is ready. You can download it or copy the content.
                  </CardDescription>
                </div>
                <Button onClick={handleDownload} variant="outline" size="sm">
                  <Download className="mr-2 h-4 w-4" />
                  Download YAML
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="relative">
                <pre className="bg-muted p-4 rounded-md text-sm overflow-auto max-h-96 whitespace-pre-wrap font-mono">
                  {generatedDSL}
                </pre>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}