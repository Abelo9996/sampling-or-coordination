"""Agent topologies. Each returns the final parsed answer + the exact list of
model CallRecords (the cost). All topologies are homogeneous (one base model)
so the measured effect is the topology, not a model mix.

Signature: topo(model, task, item, params, seed) -> {answer, raw, calls}
  - `task` is the registry dict from tasks.py (prompt/extract/score).
  - `params` is the topology's compute knobs (k, N, R, M).
  - `seed` is the base seed; stochastic sub-calls derive seeds from it.
"""
from collections import Counter
from . import client

FORMAT_REMINDER = ("End your response with exactly one line: "
                   "'The answer is N.'")


def _majority(answers):
    """Most common non-None answer; None if all None."""
    vals = [a for a in answers if a is not None]
    if not vals:
        return None
    # Counter on stringified to handle floats/ints uniformly, map back.
    c = Counter(str(v) for v in vals)
    top = c.most_common(1)[0][0]
    for v in vals:
        if str(v) == top:
            return v
    return vals[0]


def _extract(task, item, text):
    """Parse an answer. Uses the task's optional per-item extractor
    (`extract_item`, e.g. MMLU-Pro items with fewer than ten options) when
    present; otherwise exactly `task["extract"](text)`, as before."""
    f = task.get("extract_item")
    return f(text, item) if f is not None else task["extract"](text)


# ----------------------------------------------------------------------------
def sa(model, task, item, params, seed):
    """Single agent, greedy."""
    p = task["prompt"](item["question"])
    r = client.generate(model, p, temperature=0.0, seed=seed)
    return {"answer": _extract(task, item, r["text"]), "raw": r["text"], "calls": [r]}


def sc(model, task, item, params, seed):
    """Single agent + self-consistency: k samples (T>0), majority vote."""
    k = int(params.get("k", 5))
    temp = float(params.get("temperature", 0.7))
    p = task["prompt"](item["question"])
    calls, answers = [], []
    for i in range(k):
        r = client.generate(model, p, temperature=temp, seed=(seed * 1000 + i))
        calls.append(r)
        answers.append(_extract(task, item, r["text"]))
    return {"answer": _majority(answers), "raw": answers, "calls": calls}


def debate(model, task, item, params, seed, trace=None):
    """Multi-agent debate: N agents, R critique rounds, then majority vote of
    final answers. No extra LLM aggregator (vote is verifiable).

    `trace` (optional dict, default None): if given, it is filled with the
    per-agent answers of every round (`rounds`: list of lists, round 0 first)
    and the per-round majorities (`majorities`). Model calls, seeds, prompts,
    and the returned value are identical with or without it."""
    N = int(params.get("N", 3))
    R = int(params.get("R", 1))
    temp = float(params.get("temperature", 0.7))
    base = task["prompt"](item["question"])
    calls = []
    # round 0: each agent answers independently
    transcripts = []  # per-agent list of raw texts
    cur_answers = []
    for a in range(N):
        r = client.generate(model, base, temperature=temp, seed=(seed * 1000 + a))
        calls.append(r)
        transcripts.append([r["text"]])
        cur_answers.append(_extract(task, item, r["text"]))
    if trace is not None:
        trace["rounds"] = [list(cur_answers)]
        trace["majorities"] = [_majority(cur_answers)]
    # critique rounds
    for rnd in range(R):
        new_answers = []
        for a in range(N):
            others = [transcripts[o][-1] for o in range(N) if o != a]
            others_blk = "\n\n".join(
                f"Solution from agent {i+1}:\n{t}" for i, t in enumerate(others))
            p = (f"{base}\nHere are other agents' solutions to the same "
                 f"problem:\n\n{others_blk}\n\nCritique them, then give your "
                 f"updated solution. {task['reminder']}")
            r = client.generate(model, p, temperature=temp,
                                seed=(seed * 1000 + (rnd + 1) * 100 + a))
            calls.append(r)
            transcripts[a].append(r["text"])
            new_answers.append(_extract(task, item, r["text"]))
        cur_answers = new_answers
        if trace is not None:
            trace["rounds"].append(list(cur_answers))
            trace["majorities"].append(_majority(cur_answers))
    return {"answer": _majority(cur_answers), "raw": cur_answers, "calls": calls}


# Personas + temperatures for the diversity-injected debate variant. The base
# model is unchanged (still homogeneous in weights); diversity is induced through
# distinct role prompts and distinct sampling temperatures, the two cheapest
# ways to make homogeneous agents behaviorally heterogeneous.
DIV_PERSONAS = [
    "You are a careful, methodical mathematician. Work step by step and "
    "double-check every arithmetic operation.",
    "You are an inventive problem-solver. Consider more than one way to approach "
    "the problem before committing to an answer.",
    "You are a rigorous skeptic. Actively look for mistakes, hidden assumptions, "
    "and edge cases in the reasoning.",
]
DIV_TEMPS = [0.3, 0.7, 1.0]


def debate_div(model, task, item, params, seed):
    """Diversity-injected debate: identical to `debate` (N agents, R critique
    rounds, majority vote of the final round) except each agent is given a
    distinct persona prefix and a distinct sampling temperature, so the agents
    are behaviorally heterogeneous while sharing one base model. This probes
    whether *injected diversity* rescues coordination relative to the homogeneous
    debate, the boundary a homogeneous-agent design cannot itself test."""
    N = int(params.get("N", 3))
    R = int(params.get("R", 1))
    base = task["prompt"](item["question"])
    personas = [DIV_PERSONAS[a % len(DIV_PERSONAS)] for a in range(N)]
    temps = [DIV_TEMPS[a % len(DIV_TEMPS)] for a in range(N)]
    calls = []
    transcripts = []
    cur_answers = []
    for a in range(N):
        p = f"{personas[a]}\n\n{base}"
        r = client.generate(model, p, temperature=temps[a], seed=(seed * 1000 + a))
        calls.append(r)
        transcripts.append([r["text"]])
        cur_answers.append(_extract(task, item, r["text"]))
    for rnd in range(R):
        new_answers = []
        for a in range(N):
            others = [transcripts[o][-1] for o in range(N) if o != a]
            others_blk = "\n\n".join(
                f"Solution from agent {i+1}:\n{t}" for i, t in enumerate(others))
            p = (f"{personas[a]}\n\n{base}\nHere are other agents' solutions to "
                 f"the same problem:\n\n{others_blk}\n\nCritique them, then give "
                 f"your updated solution. {task['reminder']}")
            r = client.generate(model, p, temperature=temps[a],
                                seed=(seed * 1000 + (rnd + 1) * 100 + a))
            calls.append(r)
            transcripts[a].append(r["text"])
            new_answers.append(_extract(task, item, r["text"]))
        cur_answers = new_answers
    return {"answer": _majority(cur_answers), "raw": cur_answers, "calls": calls}


# Heterogeneous debate: three DIFFERENT base models (not one), the setting the
# reviewer flagged as the one that could overturn the homogeneous result. Agents
# keep temperature fixed; the diversity here is genuine model heterogeneity.
HETERO_MODELS = ["qwen2.5:3b-instruct", "llama3.2:3b", "gemma2:2b"]


def debate_hetero(model, task, item, params, seed):
    """Heterogeneous multi-agent debate: N agents, each a DIFFERENT base model
    (Qwen2.5-3B, Llama-3.2-3B, Gemma-2-2B), R critique rounds, majority vote of
    the final round. The passed `model` is ignored; agent a uses HETERO_MODELS[a].
    Tests whether genuine model heterogeneity (not prompt/temperature diversity on
    one model) lets coordination beat single-model self-consistency."""
    models = list(params.get("models", HETERO_MODELS))
    N = int(params.get("N", len(models)))
    R = int(params.get("R", 1))
    temp = float(params.get("temperature", 0.7))
    base = task["prompt"](item["question"])
    calls, transcripts, cur_answers = [], [], []
    for a in range(N):
        m = models[a % len(models)]
        r = client.generate(m, base, temperature=temp, seed=(seed * 1000 + a))
        calls.append(r)
        transcripts.append([r["text"]])
        cur_answers.append(_extract(task, item, r["text"]))
    for rnd in range(R):
        new_answers = []
        for a in range(N):
            m = models[a % len(models)]
            others = [transcripts[o][-1] for o in range(N) if o != a]
            others_blk = "\n\n".join(
                f"Solution from agent {i+1}:\n{t}" for i, t in enumerate(others))
            p = (f"{base}\nHere are other agents' solutions to the same "
                 f"problem:\n\n{others_blk}\n\nCritique them, then give your "
                 f"updated solution. {task['reminder']}")
            r = client.generate(m, p, temperature=temp,
                                seed=(seed * 1000 + (rnd + 1) * 100 + a))
            calls.append(r)
            transcripts[a].append(r["text"])
            new_answers.append(_extract(task, item, r["text"]))
        cur_answers = new_answers
    return {"answer": _majority(cur_answers), "raw": cur_answers, "calls": calls}


def critic_found_errors(text):
    """The critic verdict used by the GES stop gate and by the GE traces: the
    critic accepts iff its first 200 characters say 'no error', 'no mistake',
    or 'fully correct'."""
    head = text.strip().lower()[:200]
    return not ("no error" in head or "no mistake" in head or "fully correct" in head)


def ge(model, task, item, params, seed, trace=None):
    """Generator-Evaluator (reflexion-style): generate, then R rounds of
    [evaluate -> revise]. Final answer is the last generator output.

    `trace` (optional dict, default None): if given, it receives the draft
    answer (`draft`) and per round the critic verdict and the revised answer
    (`rounds`: list of {critic_found_errors, answer}). Model calls, seeds,
    prompts, and the returned value are identical with or without it."""
    R = int(params.get("R", 2))
    temp = float(params.get("temperature", 0.7))
    base = task["prompt"](item["question"])
    calls = []
    gen = client.generate(model, base, temperature=temp, seed=(seed * 1000))
    calls.append(gen)
    cur = gen["text"]
    if trace is not None:
        trace["draft"] = _extract(task, item, cur)
        trace["rounds"] = []
    for rnd in range(R):
        crit_p = (f"Problem:\n{item['question']}\n\nA proposed solution:\n{cur}\n\n"
                  f"Carefully check the solution for errors. List any mistakes "
                  f"concisely. If it is fully correct, say 'No errors.'")
        crit = client.generate(model, crit_p, temperature=temp,
                               seed=(seed * 1000 + (rnd + 1) * 7))
        calls.append(crit)
        rev_p = (f"{base}\nA previous attempt:\n{cur}\n\nFeedback:\n{crit['text']}\n\n"
                 f"Provide an improved, correct solution. {task['reminder']}")
        rev = client.generate(model, rev_p, temperature=temp,
                              seed=(seed * 1000 + (rnd + 1) * 13))
        calls.append(rev)
        cur = rev["text"]
        if trace is not None:
            trace["rounds"].append({
                "critic_found_errors": critic_found_errors(crit["text"]),
                "answer": _extract(task, item, cur)})
    return {"answer": _extract(task, item, cur), "raw": cur, "calls": calls}


def ges(model, task, item, params, seed):
    """Generator-Evaluator WITH a stop gate (reviewer fix): identical to `ge`,
    except that if the critic reports no errors the loop stops and the current
    answer is returned, so the pipeline is not forced into revisions of
    solutions its own evaluator accepts (the correct->incorrect flip hazard).
    Cost is therefore variable: 1 + 2*(rounds actually used) calls."""
    R = int(params.get("R", 2))
    temp = float(params.get("temperature", 0.7))
    base = task["prompt"](item["question"])
    calls = []
    gen = client.generate(model, base, temperature=temp, seed=(seed * 1000))
    calls.append(gen)
    cur = gen["text"]
    for rnd in range(R):
        crit_p = (f"Problem:\n{item['question']}\n\nA proposed solution:\n{cur}\n\n"
                  f"Carefully check the solution for errors. List any mistakes "
                  f"concisely. If it is fully correct, say 'No errors.'")
        crit = client.generate(model, crit_p, temperature=temp,
                               seed=(seed * 1000 + (rnd + 1) * 7))
        calls.append(crit)
        # STOP GATE: accept the current answer if the critic finds no errors
        head = crit["text"].strip().lower()[:200]
        if "no error" in head or "no mistake" in head or "fully correct" in head:
            break
        rev_p = (f"{base}\nA previous attempt:\n{cur}\n\nFeedback:\n{crit['text']}\n\n"
                 f"Provide an improved, correct solution. {task['reminder']}")
        rev = client.generate(model, rev_p, temperature=temp,
                              seed=(seed * 1000 + (rnd + 1) * 13))
        calls.append(rev)
        cur = rev["text"]
    return {"answer": _extract(task, item, cur), "raw": cur, "calls": calls}


def hierc(model, task, item, params, seed):
    """Hierarchical planner -> SEQUENTIALLY CHAINED executors -> aggregator
    (reviewer fix): each executor receives the outputs of all preceding
    sub-steps, so sequentially dependent decompositions are executable."""
    M = int(params.get("M", 3))
    temp = float(params.get("temperature", 0.7))
    calls = []
    plan_p = (f"Problem:\n{item['question']}\n\nBreak this problem into {M} "
              f"sequential sub-steps needed to solve it. List exactly {M} "
              f"sub-steps, one per line, numbered 1..{M}. Do not solve them yet.")
    plan = client.generate(model, plan_p, temperature=temp, seed=(seed * 1000))
    calls.append(plan)
    exec_texts = []
    for i in range(M):
        prior = ("\n\n".join(exec_texts) + "\n\n") if exec_texts else ""
        ex_p = (f"Problem:\n{item['question']}\n\nPlan:\n{plan['text']}\n\n"
                f"{prior}Carry out sub-step {i+1}, using the results of the "
                f"previous sub-steps above, and report its result concisely.")
        ex = client.generate(model, ex_p, temperature=temp,
                             seed=(seed * 1000 + 100 + i))
        calls.append(ex)
        exec_texts.append(f"Sub-step {i+1} result:\n{ex['text']}")
    agg_p = (f"{task['prompt'](item['question'])}\nUse these intermediate "
             f"results:\n\n" + "\n\n".join(exec_texts) +
             f"\n\nNow give the final solution. {task['reminder']}")
    agg = client.generate(model, agg_p, temperature=0.0, seed=(seed * 1000 + 999))
    calls.append(agg)
    return {"answer": _extract(task, item, agg["text"]), "raw": agg["text"], "calls": calls}


def hier(model, task, item, params, seed):
    """Hierarchical planner -> M executors -> aggregator."""
    M = int(params.get("M", 3))
    temp = float(params.get("temperature", 0.7))
    calls = []
    plan_p = (f"Problem:\n{item['question']}\n\nBreak this problem into {M} "
              f"sequential sub-steps needed to solve it. List exactly {M} "
              f"sub-steps, one per line, numbered 1..{M}. Do not solve them yet.")
    plan = client.generate(model, plan_p, temperature=temp, seed=(seed * 1000))
    calls.append(plan)
    # executors
    exec_texts = []
    for i in range(M):
        ex_p = (f"Problem:\n{item['question']}\n\nPlan:\n{plan['text']}\n\n"
                f"Carry out sub-step {i+1} and report its result concisely.")
        ex = client.generate(model, ex_p, temperature=temp,
                             seed=(seed * 1000 + 100 + i))
        calls.append(ex)
        exec_texts.append(f"Sub-step {i+1} result:\n{ex['text']}")
    # aggregator
    agg_p = (f"{task['prompt'](item['question'])}\nUse these intermediate "
             f"results:\n\n" + "\n\n".join(exec_texts) +
             f"\n\nNow give the final solution. {task['reminder']}")
    agg = client.generate(model, agg_p, temperature=0.0, seed=(seed * 1000 + 999))
    calls.append(agg)
    return {"answer": _extract(task, item, agg["text"]), "raw": agg["text"], "calls": calls}


TOPOLOGIES = {"SA": sa, "SC": sc, "DEBATE": debate, "DEBATED": debate_div,
              "DEBATEH": debate_hetero,
              "GE": ge, "GES": ges, "HIER": hier, "HIERC": hierc}


def get_topology(name):
    return TOPOLOGIES[name]
