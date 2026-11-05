import { defaultClientConditions, defineConfig } from "vite";
import svelte from "rollup-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ command }) => ({
  plugins: [
    tailwindcss(),
    {
      ...svelte({ compilerOptions: { dev: command === "serve", hmr: command === "serve" } }),
      async handleHotUpdate({ file, modules, server }) {
        if (!file.endsWith(".svelte")) return;
        const css = server.moduleGraph.getModuleById(file.replace(/\.svelte$/, ".css"));
        if (!css) return;
        // Regenerate virtual CSS before Vite serves its hot update.
        for (const module of modules) {
          server.moduleGraph.invalidateModule(module);
          await server.transformRequest(module.url);
        }
        if (!css.importers.size) return modules;
        server.moduleGraph.invalidateModule(css);
        return [...modules, css];
      },
    },
  ],
  resolve: {
    conditions: ["svelte", ...defaultClientConditions],
    dedupe: ["svelte"],
  },
  base: "./",
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/v1": "http://localhost:1234",
      "/ui/api": "http://localhost:1234",
      "/ui/uploads": "http://localhost:1234",
      "/ui/speech": "http://localhost:1234",
    },
  },
}));
