"""Mechanistic trace for debate and GE on GSM8K: capture initial-vs-final answers,
critic verdicts, and correctness transitions. Mirrors topologies.debate/ge with
the same prompts and seeds. -> results/mechanistic.jsonl"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import client, tasks, topologies
MODEL="qwen2.5:3b-instruct"; SEED=1; N_ITEMS=80
t=tasks.get_task("gsm8k"); items=t["load"](n=N_ITEMS, split="all")
maj=topologies._majority
def correct(ans): return None
out="results/mechanistic.jsonl"
done={json.loads(l)["item_id"]+json.loads(l)["wf"] for l in open(out)} if os.path.exists(out) else set()

def debate_trace(it):
    base=t["prompt"](it["question"]); N,R=3,1; temp=0.7; transcripts=[]; init=[]
    for a in range(N):
        r=client.generate(MODEL, base, temperature=temp, seed=SEED*1000+a)
        transcripts.append([r["text"]]); init.append(t["extract"](r["text"]))
    cur=list(init)
    for rnd in range(R):
        new=[]
        for a in range(N):
            others=[transcripts[o][-1] for o in range(N) if o!=a]
            blk="\n\n".join(f"Solution from agent {i+1}:\n{x}" for i,x in enumerate(others))
            p=f"{base}\nHere are other agents' solutions to the same problem:\n\n{blk}\n\nCritique them, then give your updated solution. {t['reminder']}"
            r=client.generate(MODEL, p, temperature=temp, seed=SEED*1000+(rnd+1)*100+a)
            transcripts[a].append(r["text"]); new.append(t["extract"](r["text"]))
        cur=new
    ic=t["score"](maj(init), it["gold"]); fc=t["score"](maj(cur), it["gold"])
    corr_present_init=any(t["score"](a, it["gold"]) for a in init)
    return dict(wf="debate", item_id=it["id"], init_correct=bool(ic), final_correct=bool(fc),
                init_unique=len(set(map(str,init))), final_unique=len(set(map(str,cur))),
                correct_minority_init=bool(corr_present_init and not ic))

def ge_trace(it):
    base=t["prompt"](it["question"]); R=2; temp=0.7
    gen=client.generate(MODEL, base, temperature=temp, seed=SEED*1000); cur=gen["text"]
    init_ans=t["extract"](cur); init_c=t["score"](init_ans, it["gold"])
    rounds=[]
    prev_c=init_c
    for rnd in range(R):
        cp=f"Problem:\n{it['question']}\n\nA proposed solution:\n{cur}\n\nCarefully check the solution for errors. List any mistakes concisely. If it is fully correct, say 'No errors.'"
        crit=client.generate(MODEL, cp, temperature=temp, seed=SEED*1000+(rnd+1)*7)
        head=crit["text"].strip().lower()[:200]
        found_errors=not ("no error" in head or "no mistake" in head or "fully correct" in head)
        rp=f"{base}\nA previous attempt:\n{cur}\n\nFeedback:\n{crit['text']}\n\nProvide an improved, correct solution. {t['reminder']}"
        rev=client.generate(MODEL, rp, temperature=temp, seed=SEED*1000+(rnd+1)*13); cur=rev["text"]
        new_c=t["score"](t["extract"](cur), it["gold"])
        rounds.append(dict(critic_found_errors=bool(found_errors), before_correct=bool(prev_c), after_correct=bool(new_c)))
        prev_c=new_c
    return dict(wf="ge", item_id=it["id"], init_correct=bool(init_c), final_correct=bool(prev_c), rounds=rounds)

with open(out,"a") as f:
    for i,it in enumerate(items):
        for tr,wf in [(debate_trace,"debate"),(ge_trace,"ge")]:
            if it["id"]+wf in done: continue
            rec=tr(it); f.write(json.dumps(rec)+"\n"); f.flush()
        print(f"  {i+1}/{len(items)} {it['id']} done", flush=True)
print("MECH DONE", flush=True)
