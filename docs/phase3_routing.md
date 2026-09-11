# Routing reference

This document is retained as a short pointer for the routing implementation.
The current architecture is described in
[`code_walkthrough.md`](code_walkthrough.md).

`ModelRouter` first validates explicit model and tier choices. Without an
override, the classifier recognizes a small task grammar and the configured
profile decides whether that task is eligible for the local model. The router
returns premium for unknown tasks, disabled profile tasks, profile/model
mismatches, and requests outside the local task range. Explicit local remains
available subject to configured admission limits.

The profile is routing policy, not a model-quality approval. A premium
provider that is missing or unhealthy returns a sanitized `503`; clients must
explicitly select local if they want local generation.
