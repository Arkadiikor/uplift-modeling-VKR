from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

SMS_COST = 9.0
PUSH_COST = 0.0
GP_NO_DISCOUNT = 5000.0
GP_SMS7 = 4300.0

@dataclass(frozen=True)
class PlanSummary:
    feasible: bool
    reason: str
    n_treatment: int
    n_control: int
    expected_profit: float
    segments: dict[str, int]
    sms_cost_estimate: float

@dataclass(frozen=True)
class CalcResult:
    feasible: bool
    reason: str
    campaign_id: str
    target_profit: float
    plans: dict[str, PlanSummary]
    sms_unit_cost: float = SMS_COST
    push_unit_cost: float = PUSH_COST
    output_dir: str | None = None

class ActionModels:
    def __init__(self, control: CatBoostClassifier, by_action: dict[str, CatBoostClassifier], feature_cols: list[str], model_version: str):
        self.control = control
        self.by_action = by_action
        self.feature_cols = feature_cols
        self.model_version = model_version

    def predict_probs(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = df[self.feature_cols]
        out = {"NONE": self.control.predict_proba(X)[:, 1]}
        for a, m in self.by_action.items():
            out[a] = m.predict_proba(X)[:, 1]
        return out

def load_action_models(model_dir: str | Path) -> ActionModels:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    model_version = meta.get("model_version", "v1")

    control = CatBoostClassifier()
    control.load_model(str(model_dir / "control.cbm"))

    by_action: dict[str, CatBoostClassifier] = {}
    mapping = {"SMS_7": "sms_7.cbm", "SMS_REM": "sms_rem.cbm", "PUSH": "push.cbm"}
    for action, fn in mapping.items():
        p = model_dir / fn
        if p.exists():
            m = CatBoostClassifier()
            m.load_model(str(p))
            by_action[action] = m

    return ActionModels(control=control, by_action=by_action, feature_cols=feature_cols, model_version=model_version)

def _delta_profit(p0: np.ndarray, pj: np.ndarray, gpj: float, cost: float) -> np.ndarray:
    return pj * gpj - p0 * GP_NO_DISCOUNT - cost

def _allowed_actions(consent_sms: int, consent_push: int, available: set[str]) -> list[str]:
    allowed = ["NONE"]
    if consent_push == 1 and "PUSH" in available:
        allowed.append("PUSH")
    if consent_sms == 1:
        if "SMS_REM" in available:
            allowed.append("SMS_REM")
        if "SMS_7" in available:
            allowed.append("SMS_7")
    return allowed

def score_best_action(feat: pd.DataFrame, probs: dict[str, np.ndarray], available_actions: set[str]) -> pd.DataFrame:
    p0 = probs["NONE"]
    dp = {}
    if "PUSH" in probs and "PUSH" in available_actions:
        dp["PUSH"] = _delta_profit(p0, probs["PUSH"], gpj=GP_NO_DISCOUNT, cost=PUSH_COST)
    if "SMS_REM" in probs and "SMS_REM" in available_actions:
        dp["SMS_REM"] = _delta_profit(p0, probs["SMS_REM"], gpj=GP_NO_DISCOUNT, cost=SMS_COST)
    if "SMS_7" in probs and "SMS_7" in available_actions:
        dp["SMS_7"] = _delta_profit(p0, probs["SMS_7"], gpj=GP_SMS7, cost=SMS_COST)

    out = feat[["customer_id", "region_code", "phone_e164", "consent_sms", "consent_push"]].copy()
    best_action = []
    best_dp = []

    for i in range(len(out)):
        cs = int(out.iloc[i]["consent_sms"])
        cp = int(out.iloc[i]["consent_push"])
        allowed = _allowed_actions(cs, cp, available_actions)

        a_best = "NONE"
        v_best = 0.0
        for a in allowed:
            if a == "NONE":
                continue
            v = float(dp[a][i])
            if v > v_best:
                v_best = v
                a_best = a

        if v_best <= 0:
            a_best = "NONE"
            v_best = 0.0

        best_action.append(a_best)
        best_dp.append(v_best)

    out["best_action"] = best_action
    out["best_delta_profit"] = best_dp
    return out

def _segments_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    vc = df[col].value_counts().to_dict()
    return {"SMS_7": int(vc.get("SMS_7", 0)), "SMS_REM": int(vc.get("SMS_REM", 0)), "PUSH": int(vc.get("PUSH", 0)), "NONE": int(vc.get("NONE", 0))}

def _stratified_split(sel: pd.DataFrame, n_control: int, n_treatment: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    df = sel.copy()

    strata_counts = df["best_action"].value_counts().to_dict()
    total = len(df)
    ctrl_by = {s: int(round(n_control * (cnt / total))) for s, cnt in strata_counts.items()}

    diff = n_control - sum(ctrl_by.values())
    if diff != 0 and strata_counts:
        s_max = max(strata_counts, key=strata_counts.get)
        ctrl_by[s_max] = max(0, ctrl_by[s_max] + diff)

    t_rows, c_rows = [], []
    for s in strata_counts.keys():
        sub = df[df["best_action"] == s].copy()
        idx = np.arange(len(sub))
        rng.shuffle(idx)

        n_c = min(ctrl_by.get(s, 0), len(sub))
        c_idx = idx[:n_c]
        t_idx = idx[n_c:]

        c = sub.iloc[c_idx].copy()
        t = sub.iloc[t_idx].copy()

        c["assigned_group"] = "control"
        c["action_type"] = "NONE"
        c["control_stratum"] = s

        t["assigned_group"] = "treatment"
        t["action_type"] = s
        t["control_stratum"] = s

        c_rows.append(c)
        t_rows.append(t)

    control = pd.concat(c_rows, ignore_index=True) if c_rows else df.iloc[0:0].copy()
    treatment = pd.concat(t_rows, ignore_index=True) if t_rows else df.iloc[0:0].copy()

    if len(treatment) > n_treatment:
        treatment = treatment.sample(n=n_treatment, random_state=seed)
    if len(control) > n_control:
        control = control.sample(n=n_control, random_state=seed)

    return treatment, control

def _select_min(scored: pd.DataFrame, n_total_min: int) -> tuple[bool, str, pd.DataFrame, float]:
    cand = scored[(scored["best_action"] != "NONE") & (scored["best_delta_profit"] > 0)].copy()
    cand = cand.sort_values("best_delta_profit", ascending=False)
    if len(cand) < n_total_min:
        return False, "INSUFFICIENT_BASE", cand, float(cand["best_delta_profit"].sum() if not cand.empty else 0.0)
    sel = cand.head(n_total_min)
    return True, "OK", sel, float(sel["best_delta_profit"].sum())

def _select_rec(scored: pd.DataFrame, target_profit: float, n_total_min: int) -> tuple[bool, str, pd.DataFrame, float]:
    cand = scored[(scored["best_action"] != "NONE") & (scored["best_delta_profit"] > 0)].copy()
    cand = cand.sort_values("best_delta_profit", ascending=False)
    if cand.empty:
        return False, "EMPTY_BASE", cand, 0.0
    cand["cum_profit"] = cand["best_delta_profit"].cumsum()
    if (cand["cum_profit"] < target_profit).all():
        return False, "TARGET_UNREACHABLE", cand, float(cand["best_delta_profit"].sum())
    k = int(np.argmax(cand["cum_profit"].to_numpy() >= target_profit)) + 1
    k = max(k, n_total_min)
    if k > len(cand):
        return False, "INSUFFICIENT_BASE", cand, float(cand["best_delta_profit"].sum())
    sel = cand.head(k).drop(columns=["cum_profit"])
    return True, "OK", sel, float(sel["best_delta_profit"].sum())

def _select_max(scored: pd.DataFrame, n_total_min: int) -> tuple[bool, str, pd.DataFrame, float]:
    cand = scored[(scored["best_action"] != "NONE") & (scored["best_delta_profit"] > 0)].copy()
    cand = cand.sort_values("best_delta_profit", ascending=False)
    if len(cand) < n_total_min:
        return False, "INSUFFICIENT_BASE", cand, float(cand["best_delta_profit"].sum() if not cand.empty else 0.0)
    return True, "OK", cand, float(cand["best_delta_profit"].sum())

def _plan_summary(sel: pd.DataFrame, expected_profit: float, n_total_min: int, seed: int) -> PlanSummary:
    n_control = n_total_min // 2
    n_treatment = n_total_min - n_control
    treatment, control = _stratified_split(sel, n_control=n_control, n_treatment=n_treatment, seed=seed)
    seg = _segments_counts(treatment, "action_type")
    n_sms = seg["SMS_7"] + seg["SMS_REM"]
    sms_cost_est = float(n_sms) * SMS_COST
    return PlanSummary(True, "OK", int(len(treatment)), int(len(control)), float(expected_profit), seg, float(sms_cost_est))

def write_outputs(outputs_dir: str | Path, campaign_id: str, plan_name: str, treatment: pd.DataFrame, control: pd.DataFrame, meta: dict) -> str:
    out_dir = Path(outputs_dir) / "campaigns" / campaign_id / plan_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for action, sub in treatment.groupby("action_type"):
        fn = f"treatment_{action.lower()}.csv"
        cols = ["customer_id", "phone_e164", "region_code", "action_type", "assigned_group", "best_delta_profit", "control_stratum"]
        keep = [c for c in cols if c in sub.columns]
        sub[keep].to_csv(out_dir / fn, index=False, encoding="utf-8")

    cols_c = ["customer_id", "phone_e164", "region_code", "action_type", "assigned_group", "control_stratum"]
    keepc = [c for c in cols_c if c in control.columns]
    control[keepc].to_csv(out_dir / "control.csv", index=False, encoding="utf-8")

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_dir)

def calc_plans(region: str, target_profit: float, models_dir: str | Path, data_dir: str | Path, n_total_min: int, seed: int) -> tuple[pd.DataFrame, dict[str, tuple[bool, str, pd.DataFrame, float]]]:
    data_dir = Path(data_dir)
    feat_path = data_dir / "marts" / "features_current.csv"
    feat = pd.read_csv(feat_path)
    feat = feat[feat["region_code"] == region].copy()

    model_dir = Path(models_dir) / "nba" / f"region={region}"
    am = load_action_models(model_dir)

    probs = am.predict_probs(feat)
    available_actions = set(am.by_action.keys()) | {"NONE"}
    scored = score_best_action(feat, probs, available_actions)

    plans_raw = {"min": _select_min(scored, n_total_min), "rec": _select_rec(scored, target_profit, n_total_min), "max": _select_max(scored, n_total_min)}
    return scored, plans_raw

def run_calc_or_confirm(region: str, target_profit: float, models_dir: str | Path, data_dir: str | Path, outputs_dir: str | Path, n_total_min: int, seed: int, dry_run: bool, confirm: bool, plan: str | None) -> CalcResult:
    campaign_id = f"{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    feat_path = Path(data_dir) / "marts" / "features_current.csv"
    if not feat_path.exists():
        return CalcResult(False, "MISSING_FEATURES", campaign_id, float(target_profit), plans={})

    model_dir = Path(models_dir) / "nba" / f"region={region}"
    if not model_dir.exists():
        return CalcResult(False, "MISSING_MODELS", campaign_id, float(target_profit), plans={})

    scored, plans_raw = calc_plans(region, target_profit, models_dir, data_dir, n_total_min, seed)

    plans: dict[str, PlanSummary] = {}
    for name, (ok, reason, sel, exp_profit) in plans_raw.items():
        if not ok:
            plans[name] = PlanSummary(False, reason, 0, 0, float(exp_profit), _segments_counts(scored, "best_action"), 0.0)
        else:
            plans[name] = _plan_summary(sel, exp_profit, n_total_min, seed)

    feasible_any = any(p.feasible for p in plans.values())
    reason = "OK" if feasible_any else "NO_FEASIBLE_PLAN"

    output_dir = None
    if confirm:
        if plan not in ("min", "rec", "max"):
            return CalcResult(False, "MISSING_PLAN", campaign_id, float(target_profit), plans=plans)
        ok, r, sel, exp_profit = plans_raw[plan]
        if not ok:
            return CalcResult(False, f"PLAN_{plan.upper()}_{r}", campaign_id, float(target_profit), plans=plans)

        n_control = n_total_min // 2
        n_treatment = n_total_min - n_control
        treatment, control = _stratified_split(sel, n_control=n_control, n_treatment=n_treatment, seed=seed)

        meta = {"campaign_id": campaign_id, "region": region, "plan": plan, "created_dt": datetime.now().isoformat(timespec="seconds"), "horizon_days": 11, "target_profit": float(target_profit), "expected_profit": float(exp_profit), "n_treatment": int(len(treatment)), "n_control": int(len(control)), "sms_unit_cost": SMS_COST, "push_unit_cost": PUSH_COST, "segments": plans[plan].segments, "sms_cost_estimate": plans[plan].sms_cost_estimate}
        output_dir = write_outputs(outputs_dir, campaign_id, plan, treatment, control, meta)

    return CalcResult(feasible_any, reason, campaign_id, float(target_profit), plans=plans, output_dir=output_dir)
