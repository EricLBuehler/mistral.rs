// =============================================================
// PMPO Artifact Refiner
// React + shadcn/ui Starter Template
// Used by UIRefinerModule for React-based artifacts
// =============================================================

import React from "react";
import { createRoot } from "react-dom/client";

// shadcn/ui components (assumes they are installed in the project)
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// =============================================================
// App Component
// =============================================================

function App() {
  return (
    <div className="min-h-screen bg-background text-foreground p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">
            {{APP_TITLE}}
          </h1>
          <p className="text-muted-foreground">
            {{APP_SUBTITLE}}
          </p>
        </header>

        {/* Example Card */}
        <Card className="rounded-2xl shadow-sm">
          <CardHeader>
            <CardTitle>Example Component</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              This React artifact was generated and refined via the PMPO Artifact Refiner.
            </p>
            <Button onClick={() => alert("Action triggered") }>
              Primary Action
            </Button>
          </CardContent>
        </Card>

        {/* Dynamic Content Slot */}
        <div id="artifact-root" className="space-y-6">
          {/* {{DYNAMIC_COMPONENTS}} */}
        </div>

      </div>
    </div>
  );
}

// =============================================================
// Mount
// =============================================================

const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
