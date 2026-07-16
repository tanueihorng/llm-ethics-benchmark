"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const REGISTRY_PATH = path.join(ROOT, "generated", "claim_registry.json");

function sha256(buffer) {
  return crypto.createHash("sha256").update(buffer).digest("hex");
}

function loadClaimRegistry() {
  if (!fs.existsSync(REGISTRY_PATH)) {
    throw new Error("Missing generated/claim_registry.json; run `make claim-registry`.");
  }
  const registry = JSON.parse(fs.readFileSync(REGISTRY_PATH, "utf8"));
  for (const [rel, expected] of Object.entries(registry.sources)) {
    const source = path.join(ROOT, rel);
    if (expected === null) {
      if (fs.existsSync(source)) {
        throw new Error(`Optional claim source is now present: ${rel}; run \`make claim-registry\`.`);
      }
      continue;
    }
    if (!fs.existsSync(source)) {
      throw new Error(`Claim source is missing: ${rel}; run \`make claim-registry\`.`);
    }
    const actual = sha256(fs.readFileSync(source));
    if (actual !== expected) {
      throw new Error(`Claim registry is stale for ${rel}; run \`make claim-registry\`.`);
    }
  }
  return registry;
}

module.exports = { loadClaimRegistry };
