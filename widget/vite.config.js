import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import { renameSync } from "node:fs";
import { resolve } from "node:path";

/** After the single-file build, expose the bundle as ``reply.html``. */
function renameToReply() {
  return {
    name: "rename-to-reply",
    closeBundle() {
      const dir = resolve(__dirname, "../app/static");
      renameSync(resolve(dir, "index.html"), resolve(dir, "reply.html"));
    },
  };
}

export default defineConfig({
  plugins: [viteSingleFile(), renameToReply()],
  build: {
    outDir: "../app/static",
    emptyOutDir: false,
    target: "esnext",
    rollupOptions: {
      input: "index.html",
    },
  },
});