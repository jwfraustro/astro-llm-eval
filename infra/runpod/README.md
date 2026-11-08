# Running open-weight models on Runpod

Only inference runs on the GPU pod. Sandboxes and the verifier run on the
control box (your laptop or any cheap CPU VM), which is where `inspect eval`
is launched.

## Flow

1. Create a pod with a PyTorch/CUDA template and enough VRAM for the model
   (a 70B at fp8 fits on one 80 GB card; a 32B in bf16 does too). Expose HTTP
   port 8000.
2. Copy `start_vllm.sh` onto the pod and run it with `MODEL` and `PARSER` set.
   Look up `PARSER` in `parsers.yaml`. Wait for "Uvicorn running".
3. On the control box:

   ```bash
   export VLLM_BASE_URL="https://<pod-id>-8000.proxy.runpod.net/v1"
   export VLLM_API_KEY="inspectai"   # or whatever API_KEY you set on the pod
   ```

   ```bash
   scripts/run.sh "vllm/Qwen/Qwen3-32B" search 5
   ```

4. Stop the pod when the run finishes; dedicated pods bill while idle.

## Things that will bite

- **Parser mismatch** looks exactly like a model that cannot use tools. Check
  one transcript in `inspect view` before launching a big run.
- **`max_model_len`** must comfortably exceed the transcript length; a
  30-message agent loop with tool output can pass 20k tokens.
- **Serverless vLLM** is tempting for cost but cold starts dominate a workload
  made of many short-ish trials. Dedicated pods are simpler for batch runs.
- Record `vllm` version, parser, quantisation, and HF revision alongside the
  results; they are part of the system under test.
