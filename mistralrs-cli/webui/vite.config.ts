import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [tailwindcss(), svelte()],
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
});
