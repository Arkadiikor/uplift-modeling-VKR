from __future__ import annotations
import argparse
import json
import sys
from src.recommender.service_multi import run_calc_or_confirm, SMS_COST, PUSH_COST

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--target_profit", required=True, type=float)
    p.add_argument("--models_dir", required=True, default="models")
    p.add_argument("--data_dir", required=True, default="data")
    p.add_argument("--outputs_dir", required=True, default="outputs")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--plan", choices=["min", "rec", "max"], default=None)
    p.add_argument("--min_total_ab", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main() -> int:
    args = parse_args()
    confirm = bool(args.confirm) and not bool(args.dry_run)
    res = run_calc_or_confirm(
        region=args.region,
        target_profit=args.target_profit,
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        n_total_min=args.min_total_ab,
        seed=args.seed,
        dry_run=bool(args.dry_run),
        confirm=confirm,
        plan=args.plan,
    )
    payload = {
        "feasible": res.feasible,
        "reason": res.reason,
        "campaign_id": res.campaign_id,
        "target_profit": res.target_profit,
        "sms_unit_cost": SMS_COST,
        "push_unit_cost": PUSH_COST,
        "plans": {
            name: {
                "feasible": p.feasible,
                "reason": p.reason,
                "n_treatment": p.n_treatment,
                "n_control": p.n_control,
                "expected_profit": p.expected_profit,
                "segments": p.segments,
                "sms_cost_estimate": p.sms_cost_estimate,
            }
            for name, p in res.plans.items()
        },
    }
    if res.output_dir:
        payload["output_dir"] = res.output_dir
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
