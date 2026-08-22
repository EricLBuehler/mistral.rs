OSV vulnerability scan evidence - mistral.rs workspace
Date: 2026-08-16
Method: osv__osv_map_dependencies on /data/mistral.rs (reads Cargo.lock, OSV batch API)
Baseline: 804 packages scanned, 13 vulnerable packages (18 advisory ids)

Fixes applied (cargo update --precise, Cargo.lock only, no manifest changes):

1. anyhow 1.0.100 -> 1.0.103
   RUSTSEC-2026-0190: unsoundness in Error::downcast_mut after Error::context.
   Direct dependency. Patch-level semver-compatible bump.

2. crossbeam-epoch 0.9.18 -> 0.9.20
   RUSTSEC-2026-0204: invalid pointer dereference in fmt::Pointer for Atomic/Shared.
   Transitive. Patch-level bump.

3. memmap2 0.9.9 -> 0.9.11
   RUSTSEC-2026-0186: unchecked pointer offset in advise_range/flush_range.
   Transitive. Patch-level bump.

4. quinn-proto 0.11.14 -> 0.11.15
   GHSA-4w2j-m93h-cj5j / RUSTSEC-2026-0185 (CVE-2026-25800): remote memory
   exhaustion via out-of-order stream reassembly. Transitive. Patch-level bump.

5. rand 0.8.5 -> 0.8.6
   GHSA-cq8v-f236-94qc / RUSTSEC-2026-0097: unsound with custom logger calling
   rand::rng() plus reseed plus trace logging. Transitive. Patch-level bump.

6. tar 0.4.45 -> 0.4.46
   GHSA-3pv8-6f4r-ffg2: PAX header desynchronization on crafted archives.
   Transitive. Patch-level bump.

Re-scan after fixes: 800 packages scanned, 8 vulnerable packages remain.

Accepted risk (no fix applied, reasoning per package):

1. aws-lc-sys 0.37.0 (10 advisories; records read: GHSA-394x-vwmw-crm3 CN
   name-constraints bypass, GHSA-65p9-r9h6-22vj AES-CCM tag timing side
   channel, GHSA-9f94-5g5w-gf6r CRL distribution point logic error).
   All three are TLS/certificate-path issues in the crypto library behind
   rustls. This deployment serves plain HTTP on 0.0.0.0:8891 with no TLS
   listener, so the rustls path is not exercised. Fix requires aws-lc-sys
   0.39.0, a minor-line jump that ripples through aws-lc-rs/rustls in the
   lockfile and diverges from the upstream-pinned tree. Accepted: TLS not
   reachable in this deployment.

2. pyo3 0.25.1 (RUSTSEC-2026-0176 OOB read in PyList/PyTuple iterators;
   GHSA-chgr-c6px-7xpp missing Sync bound on PyCFunction closures).
   pyo3 is compiled only into mistralrs-pyo3 (the Python SDK crate), which
   is not part of the mistralrs-cli binary built in this session. Fix
   (0.29.0) is a four-minor upgrade. Accepted: crate not in the deliverable.

3. core2 0.4.0 (RUSTSEC-2026-0105): unmaintained, all versions yanked, no
   fix version. Transitive. Accepted.

4. fxhash 0.2.1 (RUSTSEC-2025-0057): unmaintained, no fix version.
   Transitive. Accepted.

5. number_prefix 0.4.0 (RUSTSEC-2025-0119): unmaintained, no fix version.
   Transitive (via comfy-table). Accepted.

6. paste 1.0.15 (RUSTSEC-2024-0436): unmaintained, no fix version.
   Build-time proc macro. Accepted.

7. proc-macro-error2 2.0.1 (RUSTSEC-2026-0173): unmaintained, no fix
   version. Build-time proc macro. Accepted.

Notes:
- Cargo.lock is modified relative to upstream; revert with
  git checkout -- Cargo.lock if upstream tracking is preferred.
- Re-scan before the next release and whenever the lockfile changes;
  OSV.dev only knows advisories published so far.
- Re-scan 2026-08-16 (after the cuda_topk fix, no lockfile change):
  800 packages scanned, 7 vulnerable packages (10 advisory ids) -
  exactly the accepted-risk set listed above. No new findings.
- Re-scan 2026-08-16 (after Units 1-2 of feat/qwen35-native-context-fix:
  recurrent snapshot host parking + 2048 prefill chunk + flash-attn build;
  no lockfile change - flash-attn uses existing workspace members):
  800 packages scanned, 7 vulnerable packages (10 advisory ids) - same
  accepted-risk set. No new findings.
