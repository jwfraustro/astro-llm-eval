# Running open-weight models on Runpod

Only inference runs on the GPU pod. Sandboxes and the verifier run on the
control box (your laptop or any cheap CPU VM), which is where `inspect eval`
is launched.

## Flow

1. Create a pod with a PyTorch/CUDA template and enough VRAM for the model
   (a 70B at fp8 fits on one 80 GB card; a 32B in bf16 does too). Expose HTTP
   port 8000.
2. Copy `start_vllm.sh` onto the pod and run it with `MODEL` and `PARSER` set.
   Look up `PARSER` (and `REASONING_PARSER` for thinking models) in
   `parsers.yaml`. Wait for "Uvicorn running".
3. On the control box:

   ```bash
   export VLLM_BASE_URL="https://<pod-id>-8000.proxy.runpod.net/v1"
   export VLLM_API_KEY="inspectai"   # or whatever API_KEY you set on the pod
   ```

   ```bash
   .venv/bin/python infra/runpod/smoke.py Qwen/Qwen3-32B
   ```

   ```bash
   scripts/run.sh "vllm/Qwen/Qwen3-32B" search 5
   ```

4. Stop the pod when the run finishes; dedicated pods bill while idle.

## Things that will bite

- **Parser mismatch** looks exactly like a model that cannot use tools. Check
  one transcript in `inspect view` before launching a big run.
- **Thinking models without `REASONING_PARSER`** keep `<think>` in `content`.
  The tool parser then sees calls the model only considered, and every turn
  replays the full think block as history, which inflates context and slows
  each step.
- **Speed: per-stream vs aggregate.** Decode for one request is memory-bandwidth
  bound; an 8B model on an A100 lands somewhere around 80-150 tok/s per stream
  depending on quantisation and context length. A100s have no FP8 tensor cores,
  so `fp8` there is weight-only (Marlin): it halves weight traffic but is not
  the H100 fast path. The lever that matters for a batch eval is concurrency:
  raise `--max-samples`/`--max-sandboxes` and watch the server log's
  "Avg generation throughput" climb while per-stream speed barely drops.
  Inspect's own timings also include prefill of a growing agent transcript and
  the Runpod proxy round-trip, so they understate decode speed.
- **`max_model_len`** must comfortably exceed the transcript length; a
  30-message agent loop with tool output can pass 20k tokens.
- **Serverless vLLM** is tempting for cost but cold starts dominate a workload
  made of many short-ish trials. Dedicated pods are simpler for batch runs.
- Record `vllm` version, parser, quantisation, and HF revision alongside the
  results; they are part of the system under test.
