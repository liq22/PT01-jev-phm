"""CPU-only bearing simulation PoC; computed simulations, not industrial results.

Run from the existing project with this module beside jev_phm.py. No data-loader
fallback, real-data read, language-model download, or target-based selection.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, hilbert, lfilter, sosfiltfilt
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
from jev_phm import signal_features, unit_weights

CLASSES = ("healthy", "inner-ring", "outer-ring")
METHODS = ("fine_only", "joint_event", "weighted_libra", "conditional_anchor", "conditional_freeze", "rf_fine")


def orders(cfg):
    ratio = cfg["diameter_ratio"] * np.cos(np.deg2rad(cfg["contact_angle_deg"]))
    if not 0 < ratio < 1 or cfg["balls"] < 2:
        raise ValueError("Invalid declared bearing geometry")
    return cfg["balls"] / 2 * (1 + ratio), cfg["balls"] / 2 * (1 - ratio)


def impulse_response(fs, fn, damping, duration=0.1):
    """Causal underdamped response, arbitrary acceleration scale, not a digital twin."""
    if not 0 < fn < fs / 2 or not 0 < damping < 1:
        raise ValueError("Invalid resonance or damping")
    t = np.arange(int(fs * duration)) / fs
    return np.exp(-2 * np.pi * damping * fn * t) * np.sin(2 * np.pi * fn * np.sqrt(1 - damping**2) * t)


def generate_unit(seed, domain_id, index, label, cfg, domain, snr, mismatch=False):
    """One independent virtual unit; all windows and perturbations share its ID."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, domain_id, label, index]))
    fs, w, nw = cfg["sample_rate"], cfg["window_samples"], cfg["windows_per_unit"]
    guard = int(0.2 * fs)
    n = w * nw
    t = (np.arange(n + 2 * guard) - guard) / fs
    fr = rng.uniform(*domain["shaft_hz_range"])
    fn = rng.uniform(*domain["resonance_hz"])
    damping = rng.uniform(*cfg["damping_range"])
    amp = rng.uniform(*cfg["impact_amplitude_range"])
    gain = rng.uniform(*cfg["gain_range"])
    phase = rng.uniform(0, 2*np.pi)
    shift = cfg["mismatch"] if mismatch else {}
    ramp = shift.get("speed_ramp_fraction", 0.)
    rate = fr * (1 + ramp * (2 * t / (n / fs) - 1))
    cycles = np.cumsum(rate) / fs
    h = impulse_response(fs, fn, damping)
    if mismatch:
        h = h + shift["second_mode_weight"] * impulse_response(fs, shift["second_mode_hz"], damping)
    impulses = np.zeros(len(t))
    event_times = np.empty(0)
    oi, oo = orders(cfg)
    if label:
        order = oi if label == 1 else oo
        jitter = shift.get("interval_jitter", cfg["interval_jitter"])
        # Jitter on consecutive intervals rather than independently moving ordered events.
        count = int((cycles[-1] - cycles[0]) * order * 1.5) + 50
        intervals = (1 + rng.uniform(-np.sqrt(3)*jitter, np.sqrt(3)*jitter, count)) / order
        phases = cycles[0] + rng.uniform(0, 1 / order) + np.cumsum(intervals)
        phases = phases[phases < cycles[-1]]
        event_times = np.interp(phases, cycles, t)
        a = amp * rng.lognormal(-0.5*0.1**2, 0.1, len(phases))
        if label == 1:
            # Load-zone/direction modulation of an inner-ring defect, no oracle input.
            a *= 1 + cfg["inner_modulation"] * np.cos(2*np.pi*phases + phase)
        loc = (event_times - t[0]) * fs
        left = np.floor(loc).astype(int)
        frac = loc - left
        good = left + 1 < len(t)
        np.add.at(impulses, left[good], a[good] * (1-frac[good]))
        np.add.at(impulses, left[good]+1, a[good] * frac[good])
    # Non-periodic impacts are present in every class with the same distribution.
    nuisance = np.zeros(len(t))
    count = rng.poisson(cfg["random_impact_rate"] * len(t) / fs)
    loc = rng.integers(0, len(t), size=count)
    np.add.at(nuisance, loc, rng.normal(0, cfg["random_impact_amplitude"], count))
    response = gain * fftconvolve(impulses + nuisance, h, mode="full")[:len(t)]
    shaft = gain * (0.06*np.sin(2*np.pi*cycles+phase) + 0.03*np.sin(4*np.pi*cycles+0.7*phase))
    # Nominal reference energy is independent of the class actually simulated.
    ref_rms = gain * amp * np.sqrt(4 * fr / fs * np.sum(h*h))
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, domain_id, label, index, 919]))
    noise = lfilter([1.], [1., -cfg["noise_ar"]], noise_rng.normal(size=len(t)))
    noise *= ref_rms * 10**(-snr/20) / np.std(noise[guard:guard+n])
    clean = (response + shaft)[guard:guard+n]
    wave = clean + noise[guard:guard+n]
    rates = rate[guard:guard+n].reshape(nw, w).mean(1)
    meta = dict(seed=int(seed), domain_id=int(domain_id), index=int(index), label=int(label),
                shaft_hz=float(fr), resonance_hz=float(fn), damping=float(damping),
                gain=float(gain), impact_amplitude=float(amp), snr_reference_db=float(snr),
                mismatch=bool(mismatch), reference_rms=float(ref_rms))
    return wave.reshape(nw, w), rates, meta, event_times, clean.reshape(nw, w)


def extract_features(waves, rates, cfg):
    """Common signal representation: existing six statistics + envelope order energy.

    Only waveforms and observed mean shaft frequency enter. No latent defect label,
    generated event times, true SNR, transfer function, source ID or unit ID enters.
    """
    fs = cfg["sample_rate"]
    sos = butter(4, cfg["bandpass_hz"], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, waves, axis=-1)
    env = np.abs(hilbert(y, axis=-1)) ** 2
    env -= env.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(env * np.hanning(waves.shape[1]), axis=-1)) ** 2
    freq = np.fft.rfftfreq(waves.shape[1], 1/fs)
    lo, hi, step = cfg["order_edges"]
    edges = np.arange(lo, hi + step/2, step)
    result = []
    for x, f, p in zip(waves, rates, spec):
        o = freq / f
        energy = np.histogram(o, bins=edges, weights=p)[0]
        normalized = np.log1p(100 * energy / max(energy.sum(), 1e-30))
        result.append(np.r_[np.log1p(signal_features(x)), normalized])
    return np.asarray(result), edges


def cohort(seed, name, cfg, snr=None, mismatch=False):
    d = cfg["domains"][name]
    waves, rates, labels, units, metas = [], [], [], [], []
    for label in range(3):
        for i in range(d["units_per_class"]):
            x, f, meta, _, _ = generate_unit(seed, d["domain_id"], i, label, cfg, d,
                        cfg["source_snr_db"] if snr is None else snr, mismatch)
            unit = f"r{seed}-d{d['domain_id']}-c{label}-u{i}"
            waves.extend(x); rates.extend(f)
            labels.extend([label]*len(x)); units.extend([unit]*len(x)); metas.append({"unit":unit, **meta})
    waves, rates = np.asarray(waves), np.asarray(rates)
    x, edges = extract_features(waves, rates, cfg)
    y = np.asarray(labels, dtype=int)
    allowed = np.eye(3, dtype=bool)[y]
    if name.startswith("coarse"):
        allowed[y != 0] = [False, True, True]
    return dict(x=x, y=y, allowed=allowed, unit=np.asarray(units), waves=waves,
                rates=rates, metadata=metas, edges=edges)


def event_losses(z, allowed):
    return torch.logsumexp(z, 1) - torch.logsumexp(z.masked_fill(~allowed, -torch.inf), 1)


def weighted_libra_losses(z, allowed, eps=1e-5):
    """Port of log_prp_Loss: count factor and coefficient.detach are intentional.
    Source: agapiR/partial-label-learning/utils/utils_loss.py blob 480ea9c.
    """
    p = torch.softmax(z, 1)
    k = allowed.sum(1)
    mass = (allowed * p).sum(1)
    loss = k * torch.log(torch.clamp(1-mass, min=eps)) - (allowed * torch.log(p+eps)).sum(1)
    return (1-mass).detach() * loss


def tensor_data(data, scaler):
    xx = np.c_[scaler.transform(data["x"]), np.ones(len(data["x"]))]
    return (torch.tensor(xx, dtype=torch.float64), torch.tensor(data["allowed"], dtype=torch.bool),
            torch.tensor(unit_weights(data["unit"]), dtype=torch.float64))


def predict_logits(model, x):
    initial = x @ model["initial"]
    if model["mode"] == "freeze":
        delta = x @ model["weight"]
        return initial + torch.cat([torch.zeros_like(delta), delta, delta], dim=1)
    return x @ model["weight"]


def fit_head(mode, train, tune, cfg, ridge, initial=None, anchor=0., record=False):
    x0 = train[0][0]
    init = torch.zeros((x0.shape[1], 3), dtype=torch.float64) if initial is None else initial.clone()
    weight = (torch.zeros((x0.shape[1], 1), dtype=torch.float64) if mode == "freeze" else init.clone()).requires_grad_()
    optimizer = torch.optim.Adam([weight], lr=cfg["optimizer"]["lr"])
    best_score, best_step, best, history = np.inf, -1, None, []
    qref = None
    if len(train) == 2:
        qref = torch.softmax(train[1][0] @ init[:, 1:], 1).detach()
    started = time.perf_counter()
    for step in range(cfg["optimizer"]["steps"] + 1):
        model = {"mode":mode, "initial":init, "weight":weight}
        if step % cfg["optimizer"]["evaluate_every"] == 0:
            with torch.no_grad():
                score = sum(float(event_losses(predict_logits(model,x), m) @ w) for x,m,w in tune) / len(tune)
                if score < best_score - 1e-10:
                    best_score, best_step, best = score, step, weight.detach().clone()
                if record:
                    z = predict_logits(model, tune[-1][0]); zb = tune[-1][0] @ init
                    drift = float(torch.abs((z[:,1]-z[:,2])-(zb[:,1]-zb[:,2])).mean())
                    history.append({"step":step,"source_validation":score,"within_fault_logodds_drift":drift})
        if step == cfg["optimizer"]["steps"]:
            break
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=torch.float64)
        for pos,(x,m,w) in enumerate(train):
            z = predict_logits(model, x)
            item = weighted_libra_losses(z,m,cfg["softmax_epsilon"]) if mode == "libra" and pos == 1 else event_losses(z,m)
            loss = loss + item @ w / len(train)
            if anchor and pos == 1:
                eligible = m[:,1] & m[:,2]
                kl = (qref * (torch.log(qref.clamp_min(1e-30))-torch.log_softmax(z[:,1:],1))).sum(1)
                # Eligible mass is NOT renormalized to remove its actual source prevalence.
                loss = loss + anchor * (kl * eligible) @ w / len(train)
        # Penalize actual coefficients for free heads; residual coefficients for freeze.
        loss = loss + 0.5 * ridge * (weight[:-1]**2).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training objective; no fallback")
        loss.backward(); optimizer.step()
    return {"mode":mode,"initial":init,"weight":best}, {
        "ridge":ridge,"anchor":anchor,"validation":best_score,"selected_step":best_step,
        "executed_steps":cfg["optimizer"]["steps"],"seconds":time.perf_counter()-started,
        "last_grad_norm":float(torch.linalg.vector_norm(weight.grad)),"history":history}


def choose(mode, train, tune, cfg, initial=None):
    settings = cfg["anchor_grid"] if mode == "anchor" else [{"ridge":r,"anchor":0.} for r in cfg["ridge_grid"]]
    best, records = None, []
    for i,s in enumerate(settings):
        model, trace = fit_head(mode, train, tune, cfg, initial=initial, **s, record=True)
        records.append(trace)
        # Prefer lower validation risk; exact ties prefer lower anchor then earlier trial.
        key = (trace["validation"],s["anchor"],i)
        if best is None or key < best[0]: best = (key,model,i)
    return best[1], {"selected_trial":best[2],"trials":records}


def distribution(model, x):
    with torch.no_grad():
        return torch.softmax(predict_logits(model, x), 1).numpy()


def per_unit(p, y, units, reference):
    """Fine = coarse + conditional NLL exactly, with the same equal-unit weights."""
    p = np.clip(p, 1e-12, 1.)
    p /= p.sum(1, keepdims=True)
    reference = np.clip(reference, 1e-12, 1.)
    correct = p.argmax(1) == y
    fine = -np.log(p[np.arange(len(y)), y])
    q = np.c_[p[:,0], p[:,1:].sum(1)]
    coarse = -np.log(q[np.arange(len(y)), (y>0).astype(int)])
    conditional = fine - coarse
    drift = np.abs(np.log(p[:,1]/p[:,2])-np.log(reference[:,1]/reference[:,2]))
    brier = ((p-np.eye(3)[y])**2).sum(1)
    rows = []
    for unit in np.unique(units):
        take = units == unit
        rows.append({"unit":str(unit),"nll":float(fine[take].mean()),"coarse_nll":float(coarse[take].mean()),
              "conditional_nll":float(conditional[take].mean()),"accuracy":float(correct[take].mean()),
              "brier":float(brier[take].mean()),"logodds_drift":float(drift[take].mean())})
    return rows


def save_csv(path, rows):
    if not rows: return
    with open(path,"w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(cfg, out, resume=False):
    from sklearn.metrics import f1_score
    import joblib
    out=Path(out)
    if resume:
        if json.loads((out/"config.json").read_text()) != cfg:
            raise ValueError("Resume requires the identical saved scientific configuration")
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out/"config.json").write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(cfg["optimizer"]["threads"])
    torch.set_default_dtype(torch.float64)
    begin=time.perf_counter()
    all_rows, metrics, metadata=[],[],[]
    searches=json.loads((out/"searches.json").read_text()) if resume else {}
    def check_budget():
        if time.perf_counter()-begin>cfg["budget_seconds"]: raise TimeoutError("Declared CPU budget exhausted; keep partial outputs")
    for seed in cfg["seeds"]:
        print(f"replicate {seed}: sources", flush=True)
        sources={k:cohort(seed,k,cfg) for k in ("fine_train","coarse_train","fine_tune","coarse_tune")}
        role_units=[set(d["unit"]) for d in sources.values()]
        if any(a & b for i,a in enumerate(role_units) for b in role_units[i+1:]): raise ValueError("Unit leakage")
        scaler=StandardScaler().fit(np.vstack([sources[k]["x"] for k in ("fine_train","coarse_train")]))
        tensor={k:tensor_data(d,scaler) for k,d in sources.items()}
        f,c,ft,ct=[tensor[k] for k in ("fine_train","coarse_train","fine_tune","coarse_tune")]
        if resume and (out/f"heads_{seed}.npz").is_file() and (out/f"rf_{seed}.joblib").is_file():
            print(f"replicate {seed}: reusing completed fits", flush=True)
            saved = np.load(out/f"heads_{seed}.npz", allow_pickle=False)
            if not np.array_equal(saved["mean"],scaler.mean_) or not np.array_equal(saved["scale"],scaler.scale_):
                raise ValueError("Regenerated source preprocessing does not match saved model")
            models = {}
            for name in METHODS[:-1]:
                models[name] = {"mode":"freeze" if name=="conditional_freeze" else "event",
                    "initial":torch.tensor(saved[name+"_initial"]), "weight":torch.tensor(saved[name+"_weight"])}
            rf=joblib.load(out/f"rf_{seed}.joblib")
            if str(seed) not in searches: raise ValueError("Saved fit is missing its search record")
        else:
            models={}; searches[str(seed)]={}
            base, trace=choose("event",[f],[ft],cfg)
            models["fine_only"]=base; searches[str(seed)]["fine_only"]=trace
            for name, mode in (("joint_event","event"),("weighted_libra","libra"),("conditional_anchor","anchor"),("conditional_freeze","freeze")):
                check_budget()
                model, trace=choose(mode,[f,c],[ft,ct],cfg,initial=base["weight"])
                models[name]=model;searches[str(seed)][name]=trace
                selected=trace["trials"][trace["selected_trial"]]
                print(name, {k:selected[k] for k in ("ridge","anchor","validation","selected_step")},flush=True)
            rfs=[]; best=None
            for depth in cfg["rf_depths"]:
                for leaf in cfg["rf_min_leaf"]:
                    rf=RandomForestClassifier(n_estimators=cfg["rf_trees"],max_depth=depth,min_samples_leaf=leaf,
                            random_state=seed,n_jobs=1).fit(f[0].numpy()[:,:-1],sources["fine_train"]["y"])
                    values=[]
                    for x,m,w in (ft,ct):
                        pp=rf.predict_proba(x.numpy()[:,:-1]);mass=(pp*m.numpy()).sum(1)
                        values.append(float(-np.log(np.maximum(mass,1e-12)) @ w.numpy()))
                    score=float(np.mean(values));rfs.append({"max_depth":depth,"min_samples_leaf":leaf,"validation":score})
                    if best is None or score<best[0]:best=(score,rf,len(rfs)-1)
            rf=best[1]; searches[str(seed)]["rf_fine"]={"selected_trial":best[2],"trials":rfs}
            joblib.dump(rf,out/f"rf_{seed}.joblib")
            checkpoint={"mean":scaler.mean_,"scale":scaler.scale_}
            for name,m in models.items():
                checkpoint[name+"_weight"]=m["weight"].numpy();checkpoint[name+"_initial"]=m["initial"].numpy()
            np.savez_compressed(out/f"heads_{seed}.npz",**checkpoint)
        # All fitting/selection is complete BEFORE generating any target waveform.
        (out/"searches.json").write_text(json.dumps(searches,indent=2))
        save_sources={}
        for name,d in sources.items():
            for key in ("x","allowed","unit"):save_sources[name+"_"+key]=d[key]
            metadata.extend({"role":name,**m} for m in d["metadata"])
        np.savez_compressed(out/f"source_inputs_{seed}.npz",**save_sources)
        conditions=[(f"snr_{s}",s,False,None) for s in cfg["target_snr_db"]]
        conditions += [("mismatch",cfg["primary_snr_db"],True,None),
                       ("phase_randomized",cfg["primary_snr_db"],False,"phase"),
                       ("circular_shift",cfg["primary_snr_db"],False,"shift")]
        for name,snr,mismatch,transform in conditions:
            check_budget()
            d=cohort(seed,"target",cfg,snr,mismatch)
            if any(set(d["unit"]) & r for r in role_units):raise ValueError("Target unit leakage")
            if transform == "phase":
                freq=np.fft.rfft(d["waves"],axis=-1)
                phase_rng=np.random.default_rng(np.random.SeedSequence([seed,979]))
                angles=phase_rng.uniform(-np.pi,np.pi,size=freq.shape)
                angles[:,0]=np.angle(freq[:,0]);angles[:,-1]=np.angle(freq[:,-1])
                d["waves"]=np.fft.irfft(np.abs(freq)*np.exp(1j*angles),n=d["waves"].shape[1],axis=-1)
                d["x"],_=extract_features(d["waves"],d["rates"],cfg)
            elif transform == "shift":
                d["waves"]=np.roll(d["waves"],d["waves"].shape[1]//3,axis=1)
                d["x"],_=extract_features(d["waves"],d["rates"],cfg)
            x,_,_=tensor_data(d,scaler)
            prediction_path=out/f"predictions_{seed}_{name}.npz"
            if resume and prediction_path.is_file():
                with np.load(prediction_path,allow_pickle=False) as saved:
                    if not all(np.array_equal(saved[k],d[k]) for k in ("x","y","unit")):
                        raise ValueError("Resumed observations differ from saved predictions")
                    probs={k:saved[k].copy() for k in METHODS}
            else:
                probs={k:distribution(m,x) for k,m in models.items()}
                probs["rf_fine"]=rf.predict_proba(x.numpy()[:,:-1])
                archive={"y":d["y"],"unit":d["unit"],"x":d["x"],"shaft_hz":d["rates"],**probs}
                np.savez_compressed(prediction_path,**archive)
            for method,p in probs.items():
                rows=per_unit(p,d["y"],d["unit"],probs["fine_only"])
                all_rows.extend({"replicate":seed,"condition":name,"method":method,**r} for r in rows)
                metric={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!="unit"}
                metric["macro_f1"]=float(f1_score(d["y"],p.argmax(1),average="macro",sample_weight=unit_weights(d["unit"])))
                metrics.append({"replicate":seed,"condition":name,"method":method,"units":len(rows),**metric})
            if name==f"snr_{cfg['primary_snr_db']}" and seed==cfg["seeds"][0]:
                # Predetermined first virtual unit of each class, not selected by appearance.
                ix=np.array([0,cfg["domains"]["target"]["units_per_class"]*cfg["windows_per_unit"],2*cfg["domains"]["target"]["units_per_class"]*cfg["windows_per_unit"]])
                np.savez_compressed(out/"signal_examples.npz",wave=d["waves"][ix],shaft_hz=d["rates"][ix],labels=d["y"][ix],fs=cfg["sample_rate"])
            if name==f"snr_{cfg['primary_snr_db']}":metadata.extend({"role":"target",**m} for m in d["metadata"])
        save_csv(out/"metrics.csv",metrics);save_csv(out/"unit_metrics.csv",all_rows)
        save_csv(out/"virtual_units.csv",metadata)
    summary={"computed_simulation":True,"not_industrial_data":True,"conditions":{},"paired_effects":{},"seconds":time.perf_counter()-begin}
    for condition in sorted(set(r["condition"] for r in metrics)):
        summary["conditions"][condition]={}
        for method in METHODS:
            rr=[r for r in metrics if r["condition"]==condition and r["method"]==method]
            summary["conditions"][condition][method]={k:{"mean":float(np.mean([r[k] for r in rr])),"sd_between_cohorts":float(np.std([r[k] for r in rr],ddof=1))} for k in ("nll","accuracy","macro_f1","coarse_nll","conditional_nll","logodds_drift","brier")}
        paired={}
        for method in METHODS:
            if method=="joint_event":continue
            delta=[]
            for seed in cfg["seeds"]:
                a=[r for r in all_rows if r["replicate"]==seed and r["condition"]==condition and r["method"]==method]
                b=[r for r in all_rows if r["replicate"]==seed and r["condition"]==condition and r["method"]=="joint_event"]
                if [r["unit"] for r in a]!=[r["unit"] for r in b]:raise ValueError("Paired unit mismatch")
                delta.append(np.array([ra["nll"]-rb["nll"] for ra,rb in zip(a,b)]))
            rng=np.random.default_rng(cfg["bootstrap_seed"])
            # Stratify by fixed replicate: interval CONDITIONAL on these fitted models.
            boot=np.mean([rng.choice(v,(cfg["bootstrap_repetitions"],len(v)),replace=True).mean(1) for v in delta],axis=0)
            paired[method]={"mean":float(np.mean([d.mean() for d in delta])),"conditional_ci95":np.quantile(boot,[.025,.975]).tolist(),"cohort_deltas":[float(d.mean()) for d in delta]}
        summary["paired_effects"][condition]=paired
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (out/"environment.json").write_text(json.dumps({"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"cpu":platform.processor(),"threads":torch.get_num_threads()},indent=2))
    print(json.dumps(summary["conditions"][f"snr_{cfg['primary_snr_db']}"],indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    p.add_argument("--resume",action="store_true",help="Reuse completed fits only with identical config")
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    try:run(cfg,args.out,args.resume)
    except Exception as exc:
        if Path(args.out).is_dir():
            (Path(args.out)/"failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise

if __name__=="__main__":main()
