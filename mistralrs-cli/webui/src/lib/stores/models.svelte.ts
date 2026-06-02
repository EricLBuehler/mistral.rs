import type { UiModelInfo, Capabilities } from "../types";

class ModelStore {
  models = $state<UiModelInfo[]>([]);
  selectedModel = $state<string | null>(null);
  selectedModelKind = $state<string>("text");
  selectedInputModalities = $state<string[]>([]);
  selectedOutputModalities = $state<string[]>([]);
  capabilities = $state<Capabilities>({
    search_enabled: false,
    code_execution_enabled: false,
    tool_dispatch_url: null,
  });

  selectModel(name: string) {
    this.selectedModel = name;
    const model = this.models.find((m) => m.name === name);
    if (model) {
      this.selectedModelKind = model.kind;
      this.selectedInputModalities = model.input_modalities ?? [];
      this.selectedOutputModalities = model.output_modalities ?? [];
    }
  }

  get isMultimodal(): boolean {
    return this.selectedModelKind === "multimodal";
  }

  get isSpeech(): boolean {
    return this.selectedModelKind === "speech";
  }
}

export const modelStore = new ModelStore();
