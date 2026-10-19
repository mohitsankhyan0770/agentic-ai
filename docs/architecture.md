# Request lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant M as Memory
    participant P as Planner
    participant A as Agent
    participant T as Tools
    participant C as Critic

    U->>O: task
    O->>M: search(task)
    M-->>O: related past runs
    O->>P: plan(task, memories)
    P-->>O: steps[]
    loop each step (with goal + previous results)
        O->>A: run(step, context)
        loop until final_answer or step budget
            A->>T: action(args)
            T-->>A: observation / ERROR
        end
        A-->>O: step result
    end
    O->>O: synthesize (if >1 step)
    loop at most max_revisions + 1
        O->>C: review(task, answer, evidence)
        C-->>O: score, verdict, issues
    end
    O->>M: add(task, answer)
    O-->>U: answer + trace
```

## Failure handling

| Failure | Behaviour |
|---|---|
| Model returns invalid JSON | Error fed back; retried up to `json_retries` times, then `LLMError` |
| Unknown tool / bad args | `ERROR:` observation listing valid options; agent can retry |
| Tool raises | Exception captured as an observation |
| Same call repeated | Agent is told to use the earlier observation |
| Step budget exhausted | One forced "best answer from what you have" call, marked `success=False` |
| Planner fails | Task runs directly as a single step |
| Critic fails | Answer accepted as-is (noted in trace) |
| Memory fails | Run continues without memory (noted in trace) |
| Ollama unreachable | Clear message with the `ollama pull` command |
