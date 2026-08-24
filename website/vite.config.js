import { readFile } from "node:fs/promises";
import { defineConfig } from "vite";

const installers = [
  { fileName: "install.sh", source: new URL("../install.sh", import.meta.url) },
  { fileName: "install.ps1", source: new URL("../install.ps1", import.meta.url) },
];

function installerBuildAssets() {
  return {
    name: "installer-build-assets",
    apply: "build",
    async buildStart() {
      await Promise.all(
        installers.map(async ({ fileName, source }) => {
          this.emitFile({ type: "asset", fileName, source: await readFile(source) });
        }),
      );
    },
  };
}

function installerDevAssets() {
  return {
    name: "installer-dev-assets",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const path = request.url?.split("?", 1)[0];
        const installer = installers.find(({ fileName }) => path === `/${fileName}`);

        if (!installer) {
          next();
          return;
        }

        readFile(installer.source)
          .then((contents) => {
            response.statusCode = 200;
            response.setHeader("Content-Type", "text/plain; charset=utf-8");
            response.setHeader("X-Content-Type-Options", "nosniff");
            response.end(contents);
          })
          .catch(next);
      });
    },
  };
}

export default defineConfig({
  plugins: [installerBuildAssets(), installerDevAssets()],
});
