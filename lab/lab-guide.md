# Lab Notes: Agent Engineering with NeMo Agent Toolkit

**Prerequisites:** You completed the [setup steps](../README.md#setup). Your GPU instance has vLLM running, Phoenix is up, and `./ask` loads successfully with `Ready. Type a question, or 'help' for commands.`

**First:** Pull the latest code to ensure you have all updates.

```bash
cd nat-agent-lab
git pull
```

If anything is not running, start services:

```bash
bash gaia_tools/start_services.sh
./ask
```

---

## What you will learn

By the end of this lab you will be able to:

1. **Watch an agent reason in real time.** You will see the plan-act-observe loop execute live: tool calls, intermediate results, and final answers streaming in your terminal.
2. **Read execution traces.** You will open Phoenix, drill into span trees, and identify exactly where an agent succeeded or failed.
3. **Classify agent failures.** You will learn the three failure phases ([Failure], [Drift], [Saturation]) and match them to what you see in traces.
4. **Compare architectures on the same task.** You will run single, multi-agent, and ultrafast agents on identical questions and measure the differences in traces.
5. **Improve an agent through config changes.** You will copy a config, make a targeted edit based on trace evidence, and verify the improvement.
6. **Benchmark and compete.** You will submit your agent to a public leaderboard and see where you stand.

The lab is structured in six parts (~60 min of guided work, plus open-ended benchmarking). Each part builds on the previous one.

---

## Part 1: See the loop (5 min)

An agent is an LLM in a loop: plan, act, observe, repeat. In this part you will watch the loop run in real time.

### 1.1 Ask a question that requires a tool

```
ask> What is the tallest building in San Francisco?
```

Watch the output. The agent does not answer from memory. It calls `internet_search`, reads the results, reasons over them, and produces an answer. That sequence (LLM call → tool call → LLM call) is the plan-act-observe loop in action.

> [!TIP]
> 💡 **Lesson: agents split knowledge into two systems.** The model's weights hold parametric knowledge (frozen at training time). Tools provide non-parametric knowledge (live at inference time). This is the same paradigm shift RAG introduced for retrieval, but applied to actions. The agent trades what the model "knows" for what it can "do." That tradeoff is why agents can answer questions no static model can, but also why they are slower and less predictable.

Notice the timing printed after the response (e.g., "Thinking done (3.5s)"). This is your first latency measurement.

### 1.2 Ask a follow-up

```
ask [1]> How tall is it in meters?
```

The agent remembers your first question. You did not repeat the building name. This works because the YAML config sets `max_history: 35`, which tells NAT to append the last 35 conversation turns to the context window before each new LLM call. The prompt counter (`ask [1]>`, `ask [2]>`) tracks how many turns are in memory.

> [!NOTE]
> 📘 **Memory is a tradeoff, and not only for cost.** More history means more input tokens, which increases latency quadratically through attention. Research shows LLMs pay disproportionate attention to the beginning and end of the context window, neglecting the middle (the "lost in the middle" effect). So appending 35 turns does not mean the model *uses* all 35 equally. Past a certain point, more memory can degrade retrieval of earlier facts. `max_history` is not just a cost knob; it shapes what the model can actually recall. The sliding window here is the simplest memory strategy; more advanced approaches (summarization, retrieval-augmented, hierarchical) trade off complexity for better recall.

### 1.3 Inspect your agent

```
ask [2]> info
```

This shows the agent name (`ultrafast`), config file path, model (`MiniMaxAI/MiniMax-M2.5`), LLM parameters (`temperature=0.0, seed=42, max_tokens=16384, frequency_penalty=0.3`), and the full tool list. This is the YAML config running live.

> [!TIP]
> 💡 **Lesson: the system prompt is a program, and the LLM is its runtime.** The ultrafast agent is a `tool_calling_agent` (the "Tool Calling" pattern). It uses structured JSON `tool_calls`, not ReAct-style text reasoning. The TYPE A/B/C/D routing is baked into the `system_prompt`, not a separate orchestrator call. The key insight: the same model with a different system prompt is a fundamentally different agent. Prompt engineering is not cosmetic; it is control flow design for a probabilistic runtime. Everything you change in Part 5 is YAML, not Python, because the prompt *is* the program.

---

## Part 2: The compound probability problem (10 min)

If each step succeeds with probability p, then P(all steps pass) = p^n. In this part, you will see what that looks like in practice.

### 2.1 Run a Level 1 dev question (easy, 1-2 steps)

```
ask [2]> level dev 1, 1
```

While the agent works, watch the reasoning stream (verbose is ON by default). You will see:

- The `<think>` block where the model plans its approach and classifies the question
- Tool calls as they happen (`internet_search`, `python_executor`, etc.)
- The final `FINAL ANSWER:` line

After the agent finishes, the answer is checked automatically:

- `Expected:` the correct answer
- `Submitted:` what the agent said
- `Check: [MATCH]` or `[MISMATCH]`

Note the time and whether it got it right. Level 1 questions typically need 1-2 tool calls. With p=0.95 per step, P(success) is ~90%.

> [!WARNING]
> ⚠️ **Output format is part of the agent contract.** The benchmark scores exact-match on the text after `FINAL ANSWER:`. A correct number written as `89,706` instead of `89706`, or wrapped in extra text, scores zero. This is not a quirk of GAIA; it reflects production reality. Downstream systems (APIs, databases, UIs) parse agent output programmatically. An agent that reasons correctly but formats poorly is useless to its consumer. In production, the output schema *is* the API contract. When you see a `[MISMATCH]`, check the submitted text character by character before assuming the reasoning was wrong.

> [!WARNING]
> ⚠️ **Don't be surprised if you get a `[MISMATCH]`.** Even with `temperature=0.0` and `seed=42`, answers can vary between runs. Two things break determinism:
>
> 1. **Tensor parallelism across 8 GPUs.** vLLM shards the model across all 8 H100s. The order of floating-point reductions across GPUs is not bit-identical between runs, so small numerical differences accumulate and can flip a token choice. Determinism requires single-GPU execution, which is impractical for a 456B parameter model.
>
> 2. **Tool outputs change between runs.** If the agent calls `internet_search`, web results can differ (pages update, ranking changes). If it calls `current_datetime`, the timestamp differs. Different tool outputs mean different context for the next LLM call, which changes the entire reasoning path.

> [!TIP]
> 💡 **Lesson: one run is an anecdote; 20 runs is a measurement.** A single success proves the model *can* solve the task; it says nothing about whether it *will* solve it reliably. Agent benchmarks must report accuracy over a distribution, not cherry-picked examples. You will run systematic benchmarks in Part 6 for exactly this reason.

### 2.2 Run a Level 2 dev question (harder, more steps)

```
ask [1]> level dev 2, 1
```

This question needs more tool calls and longer reasoning. Note the time difference. More steps means more chances for the agent to pick the wrong tool, form a bad query, or lose track of its reasoning.

### 2.3 Run a few more and track results

```
ask [1]> level dev 1, 3
ask [1]> level dev 1, 5
ask [1]> level dev 2, 3
```

Keep a mental tally: how many did the agent get right? How many wrong? When it got something wrong, was it on a harder question (more steps)?

> [!TIP]
> 💡 **Lesson: the agent did not get dumber; the chain got longer.** In reliability engineering, a serial system's reliability is the product of its components. Agents are serial systems. If each step has p=0.95 success rate, 10 steps gives you 0.95^10 = 60%. There are exactly two ways to improve this: raise per-step reliability (better prompts, better tools) or reduce the number of steps (better planning, fewer unnecessary tool calls). Most practitioners reach for more powerful models first. The math says you should reach for shorter chains first.

---

## Part 3: Reading traces in Phoenix (15 min)

Phoenix captures every LLM call, tool invocation, token count, and latency using OpenTelemetry. Your agents are already traced. Now learn to read the traces.

> **Pacing guide:** Sections 3.1-3.5 are essential (do these first). Sections 3.6-3.8 are reference material you can come back to during Parts 5 and 6.

### 3.1 Open Phoenix

From a **separate terminal on your laptop**, forward port 6006:

```bash
ssh -L 6006:localhost:6006 <your-ssh-host>
```

Open [http://localhost:6006](http://localhost:6006) in your browser. You will see the project `gaia_ultrafast_agent` with all your runs from Parts 1 and 2 already captured.

### 3.2 The dashboard: what the numbers mean

The top bar shows aggregate stats across all your traces:

| Total Traces | Total Cost | Latency P50 | Latency P99 |
|---|---|---|---|
| (your count) | $0 | (median time) | (slowest time) |

**Why Total Cost shows $0:** You are running vLLM on your own GPUs, so Phoenix has no pricing table. With API-based models, this column would show real dollar costs. For self-hosted inference, use the token counts in individual spans as your proxy for cost.

> [!TIP]
> 💡 **Lesson: the gap between P50 and P99 tells you how unpredictable your agent is.** P50 (median) is the latency that half your requests finish under. P99 is the latency that 99% finish under, meaning 1 in 100 requests is slower than this. For agents, these two numbers diverge dramatically because some questions need 2 tool calls (fast) while others trigger 8+ calls with retries (slow). If your P50 is 5s but P99 is 60s, that means most questions feel snappy, but every ~100th question makes the user wait a full minute. In production, your system must handle the P99 case: timeouts, loading indicators, and retry budgets are all sized to P99, not P50. The wider the gap, the harder the agent is to deploy reliably. This gap is a direct measurement of the compound probability effect: easy questions (short chains) cluster at P50, hard questions (long chains with drift) dominate P99.

**Practitioner habit: the three-number snapshot.** Every time you open Phoenix after a set of runs, read three numbers before diving into individual traces: (1) total trace count (how many runs you have), (2) P50 latency (your typical case), (3) P99 latency (your worst case). If P99 / P50 > 5x, you have a tail latency problem worth investigating before anything else.

**Understanding the Phoenix UI:**

Each question you run through `./ask` creates one **trace**. A trace is a tree of **spans**. The top of the tree is the **root span** (the `<workflow>` that represents the entire agent turn). Inside the root, you see individual tool calls (`internet_search`, `python_executor`, `fetch_url`, etc.) as nested spans.

```
Trace = one question-answer cycle
├── <workflow>  (root span: NAT orchestration)
│   └── <workflow>  (inner span: LangChain agent execution)
│         ├── internet_search
│         ├── fetch_url
│         ├── python_executor
│         ├── python_executor
│         └── internet_search
```

The double `<workflow>` nesting is normal: the outer one is NAT's orchestration layer, the inner one is the LangChain `StateGraph` execution. Your tool calls are nested inside the inner `<workflow>`.

**Navigation tabs:**

- **Traces**: One row per question. Click any row to drill into its span tree. **Start here.**
- **Spans > Root Spans**: One row per root `<workflow>` span (same number as traces, but in flat span format). Useful if you want to sort all questions by latency.
- **Spans > All**: Every individual operation from every question, flattened into one big list. An easy question might have 5 spans; a hard question might have 15+. Combined, this list gets long fast. Use the filter bar (e.g., `name == 'internet_search'`) to isolate specific tool types. All spans show `kind: chain`; you distinguish them by **name**.
- **Sessions**: Groups multi-turn conversations (when you ask follow-ups).
- **Metrics**: Aggregate latency and token distributions across runs.

Click the **Traces** tab to see your agent turns. Each row is one complete question-answer cycle. Click any row to drill into its span tree.

### 3.3 Drill into a trace: the span tree

Click any trace. The span tree expands on the left panel to show every step the agent took. A simple question (Level 1) might look like this:

```
<workflow>                              (root: NAT orchestration)
└── <workflow>                          (LangChain agent execution)
    ├── internet_search                 (agent searched the web)
    ├── fetch_url                       (agent fetched a specific page)
    └── python_executor                 (agent ran a calculation, produced FINAL ANSWER)
```

A harder question (Level 2) will have many more spans:

```
<workflow>
└── <workflow>
    ├── internet_search                 (first search)
    ├── get_youtube_transcript          (fetched video transcript)
    ├── python_executor                 (parsed transcript)
    ├── python_executor                 (extracted data)
    ├── python_executor                 (computed intermediate result)
    ├── internet_search                 (second search for verification)
    ├── internet_search                 (third search)
    ├── fetch_url                       (fetched a source page)
    ├── python_executor                 (final computation)
    ├── python_executor                 (formatted answer)
    ...
```

The span count directly reflects the chain length. Your easy question (25s) had ~5 spans. Your hard question (2m 49s) had 15+. This is compound probability made visible.

Each span is clickable. When you click one, a detail panel opens on the right with **Info**, **Annotations**, **Attributes**, and **Events** tabs.

**What to look at in each span (Info tab):**

| Panel section | What it shows | Why it matters |
|---|---|---|
| **Input** | For tool spans: the arguments sent to the tool (e.g., search query, Python code). For workflow spans: the full message list including system prompt and conversation history. | Tool inputs reveal the quality of the agent's planning. A vague search query like `"building"` signals poor reasoning. A specific query like `"tallest building San Francisco height meters"` signals good planning. |
| **Output** | For tool spans: the tool's return value (search results, code output). For workflow spans: the model's full response including `<think>` reasoning and `FINAL ANSWER:`. | Read the Output of `python_executor` spans to verify the code ran correctly. Read the workflow Output to see the `<think>` block and final answer. |
| **Latency** | Wall-clock time for that specific span. | `internet_search` and `fetch_url` spans often take 400ms-1s (network I/O). `python_executor` spans are fast (~15-20ms). If the total trace latency is 2+ minutes, the cost is in the number of steps, not individual span speed. |
| **Status** | Green check (success) or red X (error) for this span. | A red status on a tool span means the tool failed (API timeout, rate limit, invalid input). Check if the agent retried or hallucinated around the failure. |

**Counting steps and cost:** Count the total number of tool spans in the tree. For a Level 1 question, you should see 3-6 tool calls. For Level 2, 6-12. Each tool call adds latency and its result is appended to the context for the next reasoning step. You can see the cumulative effect by clicking the root `<workflow>` span and checking the total latency and token counts in the **Attributes** tab (`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`).

> [!NOTE]
> 📘 **In production agents, input tokens are ~99% of total token cost** (in practice). This is because the model's context window reloads the full system prompt, conversation history, and all accumulated tool results before every reasoning step. Output tokens (the model's response) are comparatively small. A healthy single-turn agent uses a few thousand input tokens. If the Attributes tab shows 30K+ input tokens on a single question, the agent is likely drifting: tool results are accumulating unchecked, and the model may have already found the answer but keeps searching.

> [!TIP]
> 💡 **Lesson: context growth is the silent killer of long chains.** Every tool result is appended to the context window. By the third LLM call, you may be sending 3-5x the tokens of the first call. The cost is not just latency and dollars: transformer attention degrades on long contexts. The model pays disproportionate attention to recent tokens and the system prompt, while facts buried in the middle of accumulated tool results get neglected. This is why production agents often summarize or compress intermediate results instead of appending raw output. When you see a [Drift] failure, check whether the correct information was present in the context but ignored by the model. If so, the problem is context management, not reasoning.

### 3.4 Exercise: walk through a successful trace

Pick a trace where the agent got the right answer (green check). Click through each span from top to bottom. Answer these questions:

1. How many tool spans does the trace have? (count `python_executor`, `internet_search`, `fetch_url`, etc.)
2. What tools did it use, and in what order? Does the sequence make sense for the question?
3. Click the root `<workflow>` span. Open its **Input**. Find the `system_prompt` in the messages. This is the YAML config injected into every request.
4. Click the root `<workflow>` span. Open its **Output**. Find the `<think>` block. Did the agent classify the question (TYPE A/B/C/D) before calling a tool?
5. Still in the Output, find `FINAL ANSWER:`. This is what gets scored.
6. Click any `internet_search` span. Read the Input to see the search query the agent generated. Was it specific or vague?

**What "good" looks like in a trace (practitioner checklist):**

- ✅ **3-6 tool spans** for a Level 1 question. If you see 10+, the agent is overthinking.
- ✅ **Every tool result feeds the next step.** Click consecutive tool spans; the later ones should build on information from earlier ones, not repeat the same search.
- ✅ **`internet_search` queries are specific.** Compare the Input of each search span. Good: `"tallest building San Francisco height"`. Bad: `"building"`.
- ✅ **`python_executor` code is purposeful.** Click a `python_executor` span and read the Input `code`. It should be solving a specific subproblem, not just printing debug text.
- ✅ **No repeated tool calls** with the same or near-identical parameters.
- ✅ **`FINAL ANSWER:` is clean:** just the value, no extra text, no "The answer is..."

### 3.5 Exercise: walk through a failed trace

Now pick a trace where the agent got it wrong (`[MISMATCH]` from Part 2). If all your Part 2 answers were correct, run a harder question first: `level dev 2, 2` or `level dev 2, 3`. Level 2 questions are more likely to produce a mismatch.

**The 5-bucket failure taxonomy for tool-calling agents.** Production engineers classify agent failures into five specific buckets. As you walk through the failed trace, identify which bucket applies:

| # | Failure bucket | What you see in the trace | Example |
|---|---|---|---|
| 1 | **Tool selection** | Agent chose the wrong tool, or called no tool when one was needed. | Question requires `python_executor` for math, but agent calls `internet_search` instead. Or: agent answers from memory without any tool call. |
| 2 | **Argument generation** | Correct tool, but malformed or semantically wrong input. | `internet_search` called with a vague query like "building" instead of "tallest building San Francisco height." |
| 3 | **Tool execution** | Valid request, but the tool itself failed. | `internet_search` returns an error (API timeout, rate limit). The span shows a red X status. |
| 4 | **State / orchestration** | Tool succeeded, but the agent lost or ignored the result. | Search returned the correct answer, but the next `<think>` block does not reference it. The fact is "lost in the middle" of a growing context. |
| 5 | **Recovery amplification** | A retry or fallback made things worse. | Tool failed once, agent retried 5 times with identical queries, burning tokens and iterations without progress. |

**Walk through the trace with these questions:**

1. At which span did things go wrong? Identify the bucket number.
2. Click that tool span and read its Input. What arguments did the agent send? Were they specific or vague (bucket 2)?
3. Click the root `<workflow>` span and read the `<think>` block in the Output. Where did the reasoning diverge from what a correct answer would require? Was the correct information already in the context but ignored (bucket 4)?
4. If you see multiple tool calls with the same or near-identical parameters, that is bucket 5. Count how many redundant spans appear.

> [!TIP]
> 💡 **Lesson: bucket 4 (state/orchestration) is the hardest to diagnose and the most common in production.** The tool did its job; the model just did not use the result. This happens because tool results get buried in a growing context window. The fix is rarely "use a smarter model." It is usually "return less data from the tool" (reduce `max_results`) or "restructure the prompt to reference tool output explicitly."

### 3.6 Using filters to find patterns

Go back to the **Spans** tab. **Important:** click **"All"** (not "Root Spans") so you see every span, not just root workflows. The filter bar at the top is your most powerful diagnostic tool. Since all spans show `kind: chain` in this setup, you filter by **name** to isolate specific tool types:

**Essential filters:**

| Filter query | What it reveals |
|---|---|
| `name == 'python_executor'` | All code execution spans. Click each to read the Input `code` and Output result. Did the code solve the right problem? |
| `name == 'internet_search'` | All web search spans. Compare search queries across spans: are they getting more specific, or repeating? |
| `name == 'fetch_url'` | All URL fetch spans. These tend to be slower (400ms-1s) due to network I/O. Check if the fetched page was relevant. |
| `name == 'describe_image'` | Image analysis spans (if the question involved an image file). |
| `status_code == 'ERROR'` | All failed spans regardless of type. Starting points for bucket 3 (tool execution failure) diagnosis. |

**Practitioner pattern: the "tool echo."** Filter to `name == 'internet_search'` and read the Input for each span in chronological order. If two consecutive searches have near-identical queries (e.g., "tallest building SF" then "tallest building San Francisco"), the model is failing to recognize it already has the information. This echo pattern is the most common source of wasted tokens. The fix is usually `frequency_penalty` (nudges the model away from repeating) or a prompt rule ("Do not search for the same fact twice").

**Practitioner pattern: the "code spiral."** Filter to `name == 'python_executor'` and read the Output of each span. If you see outputs like `"Let me re-read the problem statement..."` or repeated attempts at the same calculation with minor variations, the agent is stuck in a code loop. Check how many `python_executor` spans appear in the trace. Three or fewer is healthy for a math question. Seven or more is a spiral.

> [!TIP]
> 💡 **Lesson: in RL terms, every action has a cost; wasted actions are negative reward.** Click consecutive tool spans and compare: does the second span's work build on the first span's result, or is it redundant? If a tool result is never used by the agent's subsequent reasoning, that round-trip added latency and tokens for zero value. This is the agent equivalent of an unnecessary API call in a microservice: it costs time and money, and the system would be strictly better without it. When you see wasted tool calls, the fix is usually in the system prompt (add explicit guidance on when *not* to call a tool) or in the tool list itself (remove tools the agent confuses with each other).

### 3.7 Classify failures using the diagnostic model

The lecture introduced three failure phases: **[Failure]**, **[Drift]**, **[Saturation]**. Now find them in real traces.

Open a failed trace and classify it:

| Phase | What you see in the trace | Example |
|---|---|---|
| **[Failure]** | Wrong on the first attempt. Agent hallucinated, picked the wrong tool, or formatted the answer incorrectly. | Agent skips `internet_search` and guesses from memory. Or: correct number but wrapped in extra text, failing exact-match. Open the root `<workflow>` span's Output and read the `<think>` block. |
| **[Drift]** | Starts correct but degrades over steps. Agent searches, gets results, but then searches again unnecessarily, or loses the key fact in a growing context. | First `internet_search` returns the answer, but agent runs 3 more. Span tree shows 8+ `python_executor` calls with outputs like "Let me re-read..." The context is bloated and the model is going in circles. |
| **[Saturation]** | System crashes or times out. Too many iterations, context limit hit. | Span tree has 15+ tool spans. Total trace latency exceeds 3-4 minutes. Later spans may show errors or the model produces no `FINAL ANSWER:` at all. |

**Exercise:** Find one trace in each phase (or at least two of three). For each, write down:

1. The phase label ([Failure], [Drift], or [Saturation])
2. Which tool span went wrong (by name, e.g., "3rd `internet_search`")
3. What the model was thinking (open the root `<workflow>` Output, find the `<think>` block, paste the first line)
4. What config change might fix it (you will make this change in Part 5)

> [!TIP]
> 💡 **Lesson: trace → classify → fix → re-benchmark is the agent training loop.** This is the agent equivalent of the standard ML cycle (train → eval → diagnose loss curves → iterate). The difference: you are optimizing prompts and tool configs instead of weights and hyperparameters. The discipline is identical. Without traces, you are guessing; with traces, you are engineering. The failure phase label ([Failure], [Drift], [Saturation]) tells you *where* in the config to look, just like a diverging loss curve tells you to check learning rate first.

### 3.8 Pro tips: what production engineers look for

These are patterns from practitioners who debug agents in production daily. You will not need all of them right now, but they will save you hours when you build your own agent in Part 5 and beyond.

**1. Read the `<think>` block in the workflow Output.** Click the root `<workflow>` span and open its Output. The model's reasoning chain (inside `<think>...</think>` tags) is the most information-dense part of the trace. It reveals: (a) how the model interpreted the question, (b) what it plans to do next, (c) whether it is on track or drifting. If the `<think>` block says "I already found the answer, but let me verify..." that is the beginning of a drift spiral. The model is about to waste 3-5 tool calls "verifying" something it already knows. This is the single most common failure mode in the ultrafast agent, and the prompt rule "NEVER verify an answer you are confident about" exists specifically to prevent it.

**2. Count `python_executor` spans as a health metric.** You may see the agent calling `python_executor` many times in a row with outputs like "Let me re-read the problem statement..." This is a code spiral: the model is writing Python to reason through the problem step by step instead of planning first. A healthy Level 1 question needs 1-3 `python_executor` calls. If you see 7+, the agent is stuck.

**3. Compare `internet_search` inputs chronologically.** Click each `internet_search` span in order and read the Input query. In a healthy trace, queries get more specific over time ("tallest building" then "Salesforce Tower height meters"). In a failing trace, queries are vague or repetitive ("building SF" then "SF building tall"). The quality of search queries is the most actionable diagnostic signal because you can directly improve it through the system prompt.

**4. Check latencies by span name.** `internet_search` and `fetch_url` spans typically take 400ms-1s (network-bound). `python_executor` spans are nearly instant (~15-20ms). If your total trace latency is 2 minutes but individual tool spans are fast, the cost is in the *number* of steps, not the speed of each step. This means the fix is in the prompt (fewer steps) or `max_iterations` (hard cap), not in the infrastructure.

**5. Use the Attributes tab for exact token counts.** Click any span and switch to the **Attributes** tab (next to Info). Look for `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens`. These structured fields are more reliable than eyeballing the Input/Output panels. If `input_tokens` exceeds 30K on any span, the context is bloated.

**6. Cross-reference Phoenix with terminal output.** The verbose output in your terminal (the `<think>` blocks and tool calls streaming live) shows the same information as Phoenix, but in real time. Phoenix adds what the terminal cannot: token counts, latency breakdowns per span, and the ability to compare across runs. Use the terminal to watch the agent live; use Phoenix to analyze it afterward.

> [!TIP]
> 💡 **Lesson: observability is not optional for agents; it is the only way to debug them.** Traditional software has stack traces and debuggers. Agents have traces and span trees. The non-deterministic nature of LLM reasoning means you cannot step through the code with breakpoints. The trace is your stack trace, the `<think>` block is your debugger watch window, and the token count is your performance profiler. Master these three, and you can diagnose any agent failure.

---

## Part 4: Compare architectures (10 min)

The lab ships three agents that share the same model and tools but use different control flow. All three run the same two-node LangGraph (`agent` node and `tool` node with a conditional edge). They differ in four things: system prompt, tool list, iteration budget, and orchestration structure. In this part, you will see how those differences show up in traces.

### 4.1 Run the same question on all three agents

Pick a dev question you already ran. `level dev 1, 1` is a good choice because it needs multiple tool calls, which is where architecture differences show up.

```
ask [1]> switch single
ask> level dev 1, 1

ask [1]> switch multi
ask> level dev 1, 1

ask [1]> switch ultrafast
ask> level dev 1, 1
```

After each run, note: correct/incorrect, time, and number of tool calls.

### 4.2 Compare traces side by side in Phoenix

Refresh Phoenix. You now have traces from three different agent projects. Each agent writes to its own Phoenix project (e.g., `gaia_single_agent`, `gaia_multi_agent`, `gaia_ultrafast_agent`). Use the project dropdown at the top-left to switch between them, or open multiple browser tabs.

For the same question across all three agents, record:

| Agent | Pattern | What to look for in the trace |
|---|---|---|
| **Single** | Tool Calling (flat) | Baseline. Count the tool spans. Click the root `<workflow>` span's Output to read the `<think>` block. Check whether any tool calls were repeated or wasted. Note total latency and span count. |
| **Multi** | Multi-Agent (orchestrator + specialists) | The trace will show nested workflows: the orchestrator's `<workflow>` contains specialist `<workflow>` spans inside it. Open the orchestrator's Output to see the routing decision in the `<think>` block. Was the routing correct? Compare total latency to Single for the same question. |
| **Ultrafast** | Tool Calling + prompt-driven routing | Same flat architecture as Single, but open the root `<workflow>` span's Output. You should see TYPE A/B/C/D classification in the `<think>` block before any tool call. Did the classification lead to better tool selection or fewer wasted calls? Compare span count to Single. |

> [!TIP]
> 💡 **Lesson: prompt structure is architecture, not decoration.** Single and ultrafast share the same `_type` (`tool_calling_agent`), same tools, same LLM parameters. The only difference is the system prompt. Ultrafast classifies into TYPE A/B/C/D before choosing a tool; single reasons freely. Classification before action is the prompt-level equivalent of an if/else branch vs. an open-ended while loop. The model's capabilities did not change; you changed its control flow through language. In Part 5, you will see that prompt changes routinely outperform model upgrades at a fraction of the cost.

> [!NOTE]
> 📘 **The orchestrator should be cheap and fast, like a load balancer.** In distributed systems, the scheduler must be fast and stateless; a manager that does the workers' job is an anti-pattern. The same principle applies here. The multi-agent orchestrator is deliberately constrained: `max_tokens: 1024` (vs 16384 for specialists) and `max_iterations: 5` (vs 20-30). Open the orchestrator's root span Output in Phoenix: it should be a short delegation call, not a chain of thought. If the orchestrator starts reasoning about the question instead of routing it, you have a bottleneck in the wrong place.

> [!NOTE]
> 📘 **Specialist descriptions are interface contracts written in natural language.** In traditional software, you define APIs with typed schemas. In multi-agent systems, the `description` field serves the same purpose: it is the contract the orchestrator relies on to make routing decisions. Open `multi-agent/gaia_agent_multi.yml` and read each specialist's `description`. If you changed "Expert at analyzing files and images" to "General helper," the orchestrator would misroute image questions. The quality of your descriptions directly determines routing accuracy, just as the quality of your API docs determines whether callers use your service correctly.

**Concrete comparison exercise:** For each agent, write down these three numbers from Phoenix:

1. Number of tool spans (count `python_executor`, `internet_search`, etc. in the span tree)
2. Total input tokens (click the root `<workflow>` span, go to **Attributes** tab, find `gen_ai.usage.input_tokens`)
3. Total trace latency (shown on the root `<workflow>` span)

> [!NOTE]
> 📘 **Quantify, don't guess. Intuition is unreliable for agents.** These three numbers (span count, input tokens, latency) turn "it felt faster" into a measurable comparison you can reproduce. Agent behavior is counterintuitive: the agent that "looks smarter" in its reasoning may be slower and less accurate than one that takes a direct path. The traces are ground truth. Any claim about agent performance that is not backed by traces and numbers is an opinion, not an observation.

### 4.3 Measure routing overhead on a trivial question

```
ask [1]> switch single
ask> What is the capital of France?

ask [1]> switch multi
ask> What is the capital of France?
```

This needs zero tools. Compare latencies. Multi should be noticeably slower because the orchestrator LLM call runs even when routing adds no value. This is the overhead cost of multi-agent orchestration.

> [!TIP]
> 💡 **Lesson: there is no universally best architecture (No Free Lunch).** Flat agents are faster on simple questions because they skip the routing overhead. Routing pays off on complex multi-modal questions where specialist knowledge matters. The "best" architecture depends entirely on your task distribution. If 80% of queries are simple lookups, multi-agent overhead is pure waste. If 80% require file analysis plus web search, the routing investment pays for itself. Traces make this measurable, not a matter of opinion. When someone claims "multi-agent is always better," ask to see the traces.

---

## Part 5: Improve an agent (15 min)

You have seen traces, classified failures, and compared architectures. Now fix something. Parts 5 and 6 follow eval-driven development (EDD): define what "good" means before changing anything, make one change, measure the result against your baseline, and only keep changes that pass the bar. This is the TDD of agent engineering.

### 5.1 Create your own agent

```bash
mkdir my-agent
cp ultrafast-agent/gaia_agent_ultrafast.yml my-agent/config.yml
```

Before loading, change the Phoenix project name so your custom traces are separate from the built-in ultrafast traces. Open `my-agent/config.yml` and find this line near the top:

```yaml
        project: gaia_ultrafast_agent
```

Change it to:

```yaml
        project: my_agent
```

Now load it:

```
ask> switch my-agent/config.yml
```

Your traces will now appear in a separate `my_agent` project in Phoenix.

### 5.2 Read the YAML config

```bash
cat my-agent/config.yml
```

This is the standard NAT config pattern. Key sections:

- **`workflow.system_prompt`**: The instructions the LLM sees before every question. This is where routing logic (TYPE A/B/C/D), formatting rules ("FINAL ANSWER: ..."), and tool-use guidance live. This is the most impactful thing to change.
- **`workflow.tools`**: List of available tools. Each tool's schema is injected into the prompt. Fewer tools = shorter prompt = fewer input tokens per LLM call = faster inference. Remove tools the agent never uses for your test questions.
- **`llms.nim_llm`**: LLM settings. `temperature` (0.0 = deterministic tool selection), `seed` (42, reproducibility on single GPU), `max_tokens` (16384, output cap), `frequency_penalty` (0.3, reduces repetition in long outputs).
- **`workflow.max_iterations`**: How many plan-act-observe loops the agent can run (default: 30). Too low: hard questions get cut off. Too high: drift and saturation risk.

> [!NOTE]
> 📘 **Tool error recovery is a fault-tolerance design choice.** All lab agents set `handle_tool_errors: true`. This is the agent equivalent of a circuit breaker pattern in distributed systems: when a tool throws an error, the error message is fed back to the LLM as an observation instead of crashing the chain. The model becomes the recovery mechanism, retrying with different parameters or trying a different tool. This is both powerful (graceful degradation) and risky (the model may hallucinate around the error instead of truly recovering). Set this to `false` and a single `internet_search` timeout kills the entire run. In production, you want `true` with monitoring: trace the retries and alert on repeated failures.

> [!TIP]
> 💡 **Lesson: every prompt rule is a patch for a real failure, not a preemptive guard.** Read the `system_prompt` carefully. You will find explicit anti-patterns: "NEVER answer from memory," "NEVER verify an answer you are confident about," "NEVER use internal reasoning for calculations." These are defensive programming for a probabilistic runtime. In traditional code, you add assertions for edge cases. In agent prompts, you add rules for behaviors the model drifts toward. But unlike code assertions, prompt rules are *soft* constraints: they reduce the probability of bad behavior without eliminating it. This is why you still need traces. When you write your own prompt, add rules only for failures you have diagnosed, not preemptively. An untested prompt rule is cargo-cult engineering.

> [!NOTE]
> 📘 **`frequency_penalty` is the exploration/exploitation knob for agents.** The value `0.3` penalizes tokens the model has already generated. In RL terms, without penalty the model *exploits* familiar token sequences (repeating the same search query). With penalty, it is nudged toward *exploring* different formulations. This matters far more in a 30-iteration agent loop than in a single-turn chat. Without penalty, you will see the model search "tallest building SF" five times in a row. With penalty, it tries "tallest building San Francisco height meters" on the second attempt. Compare traces from `ultrafast` (penalty 0.3) vs `single` (same penalty but different prompt structure) on the same question to see this directly.

### 5.3 Pick one thing to fix

Go back to the failure you classified in Part 3. Pick the fix that matches the failure phase:

| Failure phase | Config change | Where in YAML |
|---|---|---|
| **[Failure]**: wrong tool | Add explicit tool-selection guidance | `system_prompt` |
| **[Failure]**: bad formatting | Add "Always end with FINAL ANSWER: <value>" | `system_prompt` |
| **[Drift]**: too many searches | Add "Do not search more than twice for the same fact" | `system_prompt` |
| **[Drift]**: context bloat | Reduce `max_results` for search tools | Tool config |
| **[Saturation]**: loops forever | Lower `max_iterations` from 30 to 15 | `workflow` section |
| General: too slow | Remove tools unused by your test questions (e.g., `describe_image_alt`, `wiki_search` if `internet_search` covers it) | `tools` list |

> [!TIP]
> 💡 **Lesson: more tools means more ways to be wrong.** Each tool's schema is injected into the prompt, expanding the context and giving the model more options to pick incorrectly. With 11 tools, the model must implicitly reason about which one to use at each step. With 2, the decision is nearly trivial. In production, teams have seen that cutting the tool count from 16 to 2 raised accuracy from 80% to 100%. If your test questions only need web search and Python, remove the other 9 tools and measure the difference.

### 5.4 The iteration loop

> [!TIP]
> 💡 **Lesson: one variable at a time. Agents have more confounders than training runs.** Same discipline as hyperparameter sweeps, but harder to enforce. In training, changing learning rate and batch size simultaneously is a recognized anti-pattern. In agent engineering, the temptation is stronger: you rewrite the prompt, remove two tools, and change `max_iterations` in one edit. When accuracy changes, you cannot attribute it. Isolate, measure, decide. This is slow, but it is the only way to build reliable knowledge about what works.

1. **Define your eval**: Before touching the config, write down what "better" means for this change. For example: "fewer tool spans," "correct answer on dev 1,1," or "latency under 10s." This is the EDD principle: the eval exists before the change, not after.
2. **Before**: Note the trace timestamp in Phoenix for your current run (it is in the "Traces" table).
3. Edit `my-agent/config.yml` (one change only).
4. Run the same dev question: `level dev 1, 1`
5. **After**: Refresh Phoenix. Your new trace appears at the top of the list (most recent). Open it side by side with the old trace (use two browser tabs, one for each timestamp).
6. Compare against your eval from step 1: span count, total input tokens (Attributes tab), total latency, and correctness.
7. Pass? Keep the change. Fail? Revert (re-edit the YAML).
8. Repeat with a different change.

**Phoenix tip:** If you want to quickly find your before/after traces, filter by timestamp or just sort by "Start Time" descending. The two most recent traces in your agent's project are your before and after.

### 5.5 Test on a different question

Your change might help on one question but hurt on another. Run 2-3 different dev questions to check:

```
ask [1]> level dev 1, 3
ask [1]> level dev 2, 1
```

> [!WARNING]
> ⚠️ **Overfitting to one question is real, and it is the agent equivalent of overfitting to one training batch.** A prompt tweak that fixes question 1 may break question 3. A model that memorizes the training set fails on the test set. The same holds here. Your dev questions are your validation set; the leaderboard is your test set. Always test on 2-3 different dev questions before calling a change "good." If you only test on one question, you are memorizing, not generalizing.

If accuracy held and latency improved (or vice versa), your change is likely good. If it broke a different question, investigate that trace.

---

## Part 6: Benchmark and compete (ongoing)

### 6.1 Run the benchmark

When you are confident in your agent:

```
ask> benchmark my-agent/config.yml
```

This runs 20 scored questions using your custom config and submits to the [public leaderboard](https://huggingface.co/spaces/agents-course/Students_Leaderboard). You will be prompted for your org and team name. Your entry appears as `NAT-<org>-<team>`.

You can also type `benchmark` (with no argument) for an interactive menu that lists all built-in agents plus a custom option.

While the benchmark runs (~15 min), open Phoenix and watch the traces stream in live. Each benchmark question generates a new trace. You can observe the agent's behavior in real time.

### 6.2 Post-benchmark analysis in Phoenix

After the benchmark finishes, go to Phoenix and switch to the `my_agent` project (or whatever you named it in Part 5.1). You now have 20 traces, one per question.

**Sort by latency** (click the "Latency" column header). Your slowest questions are at the top. These are the ones most likely to have drift or saturation. Click into the top 3 slowest traces and check:

- How many tool spans in the trace?
- Did the agent loop more than necessary?
- Was the context window bloated by tool results?

**Sort by status** to see failures. Click into each failed trace. Classify it using the [Failure]/[Drift]/[Saturation] model from Part 3.

### 6.3 Three dimensions to optimize

1. **Accuracy** (leaderboard score). The built-in agents score 85-90%. Can you beat them?
2. **Speed** (total benchmark time). Check `gaia_summary.json` for per-question timing. Cross-reference with Phoenix: sort traces by latency to find the bottleneck questions. A faster agent at the same accuracy is a better agent.
3. **Design** (trace quality). Open your traces alongside a built-in agent's for the same question (different Phoenix projects). Fewer tool calls, cleaner routing, shorter prompts. Be ready to explain what you changed and why.

```bash
cat my-agent/runs/latest/gaia_summary.json    # your score
bash gaia_tools/gaia_run.sh --history          # all past runs
```

### 6.4 Recap

Everything in this lab connects to a core concept:

- **Compound probability**: you saw it in Part 2, measured it in Part 3.
- **Design patterns**: you compared three architectures in Part 4.
- **Failure phases**: you classified [Failure], [Drift], and [Saturation] in Part 3, then fixed one in Part 5.
- **The eval-driven loop**: Trace, classify, fix, re-eval. You ran it end to end.
- **Config-driven agents**: everything you changed was YAML. No Python glue code.

> [!TIP]
> 💡 **The gap between a demo and a reliable system is the engineering you just practiced.** Anyone can get an agent to answer one question correctly. The hard part is making it answer 100 questions correctly, under time constraints, with observable behavior and diagnosable failures. Tracing, failure classification, targeted config changes, and measurement across a distribution: this is what separates a prototype from a production system. The same discipline applies whether you are building an agent, a recommendation engine, or a self-driving stack. Systematic evaluation is not overhead; it is the product.

---

## Quick reference

| Command | What it does |
|---------|-------------|
| `<any text>` | Ask the agent anything (multi-turn) |
| `level dev <L>, <N>` | Run dev question with answer checking |
| `level <L>, <N>` | Run test question (no answers) |
| `benchmark` | Run 20-question scored leaderboard submission |
| `switch <agent>` | Change agent: `single`, `multi`, `ultrafast`, or a config path |
| `info` | Current agent details, model, tools |
| `status` | Service health check |
| `tracing` | Toggle Phoenix tracing (on/off/status) |
| `clear` | Reset conversation memory |
| `help` | Full command list |
| `quit` | Exit (services keep running) |

## Phoenix quick reference

| Action | How |
|--------|-----|
| Open Phoenix | `ssh -L 6006:localhost:6006 <your-ssh-host>`, then [localhost:6006](http://localhost:6006) |
| Switch agent project | Top-left dropdown (e.g., `gaia_ultrafast_agent`, `gaia_single_agent`) |
| See all traces | Click **Traces** tab |
| See all spans (not just roots) | Click **Spans** tab, then click **"All"** (not "Root Spans") |
| Filter to search calls | Type `name == 'internet_search'` in the filter bar |
| Filter to code execution | Type `name == 'python_executor'` in the filter bar |
| Find failed spans | Type `status_code == 'ERROR'` in the filter bar |
| Sort by latency | Click the **Latency** column header |
| Read model reasoning | Click the root `<workflow>` span, look at Output, find the `<think>` block |
| See tool inputs/outputs | Click any tool span (e.g., `python_executor`), check Input (code/query) and Output (result) |
| Check token counts | Click any span, switch to **Attributes** tab, find `gen_ai.usage.input_tokens` |

## Useful files

| File | What it contains |
|------|-----------------|
| `<agent>/gaia_agent*.yml` | Agent config (system prompt, tools, LLM params) |
| `<agent>/runs/latest/gaia_summary.json` | Latest benchmark score |
| `<agent>/runs/latest/gaia_results.json` | Per-question answers and correctness |
| `.env` | Your API keys (do not share) |
