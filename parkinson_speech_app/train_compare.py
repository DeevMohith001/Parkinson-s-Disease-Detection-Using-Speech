"""Train several models on the 80 speech features, compare them with nested,
repeated cross-validation, pick the best one by RECALL (subject to an accuracy
floor), and save a bundle that app.py / predict.py can load.

Usage:
    python train_compare.py --data your_data.csv --label label
    python train_compare.py --data your_data.csv --label label --group subject_id

Label column: 1 = Parkinson's (PwPD), 0 = healthy control.
Use --group if one person contributes more than one row (otherwise the same
person ends up in train AND test and every score is inflated).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # also silences joblib workers

import argparse
import json
import tempfile
import warnings
from pathlib import Path

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
FEATURES = json.loads((ROOT / "feature_names.json").read_text(encoding="utf-8"))
SEED = 42
# Yeo-Johnson is slow (~1 s/fit) and independent of every grid parameter, so cache it per fold.
MEMORY = joblib.Memory(tempfile.mkdtemp(prefix="pd_cache_"), verbose=0)


def pre(*steps):
    """Common preprocessing; everything is fitted inside CV folds (no leakage)."""
    return [("yj", PowerTransformer(standardize=False)), *steps, ("sc", StandardScaler())]


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, small hyper-parameter grid). Small grids on purpose:
    with ~80 rows, big searches just overfit the validation folds."""
    kbest = lambda: ("sel", SelectKBest(f_classif))  # noqa: E731
    k_grid = {"sel__k": [10, 20, 40]}
    cw = "balanced"
    return {
        "LogReg (L2)": (
            Pipeline(pre(kbest()) + [("clf", LogisticRegression(max_iter=5000, class_weight=cw))], memory=MEMORY),
            {**k_grid, "clf__C": [0.01, 0.1, 1]},
        ),
        "LogReg (L1)": (
            Pipeline(pre() + [("clf", LogisticRegression(penalty="l1", solver="liblinear", class_weight=cw, max_iter=5000))], memory=MEMORY),
            {"clf__C": [0.05, 0.1, 0.5, 1]},
        ),
        "SVM (RBF)": (
            Pipeline(pre(kbest()) + [("clf", SVC(probability=True, class_weight=cw, random_state=SEED))], memory=MEMORY),
            {**k_grid, "clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.01]},
        ),
        "SVM (linear)": (
            Pipeline(pre(kbest()) + [("clf", SVC(kernel="linear", probability=True, class_weight=cw, random_state=SEED))], memory=MEMORY),
            {**k_grid, "clf__C": [0.01, 0.1, 1]},
        ),
        "kNN": (
            Pipeline(pre(kbest()) + [("clf", KNeighborsClassifier())], memory=MEMORY),
            {**k_grid, "clf__n_neighbors": [3, 5, 9]},
        ),
        "LDA (shrinkage)": (
            Pipeline(pre(kbest()) + [("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))], memory=MEMORY),
            k_grid,
        ),
        "Random Forest": (
            Pipeline([("clf", RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=SEED, n_jobs=1))]),
            {"clf__max_depth": [3, None], "clf__min_samples_leaf": [1, 3]},
        ),
        "Soft-vote ensemble": (
            # Fixed, sensible members (no big grid): averaging probabilities of diverse models.
            Pipeline([("clf", VotingClassifier(voting="soft", estimators=[
                ("lr", Pipeline(pre(("sel", SelectKBest(f_classif, k=20))) + [("c", LogisticRegression(C=0.1, max_iter=5000, class_weight=cw))], memory=MEMORY)),
                ("svm", Pipeline(pre(("sel", SelectKBest(f_classif, k=20))) + [("c", SVC(C=1, probability=True, class_weight=cw, random_state=SEED))], memory=MEMORY)),
                ("lda", Pipeline(pre(("sel", SelectKBest(f_classif, k=20))) + [("c", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))], memory=MEMORY)),
                ("rf", RandomForestClassifier(n_estimators=300, max_depth=3, class_weight=cw, random_state=SEED, n_jobs=1)),
            ]))]),
            {},
        ),
        "Extra Trees": (
            Pipeline([("clf", ExtraTreesClassifier(n_estimators=300, class_weight=cw, random_state=SEED, n_jobs=1))]),
            {"clf__max_depth": [3, None], "clf__min_samples_leaf": [1, 3]},
        ),
    }


def make_inner_cv(groups):
    return StratifiedGroupKFold(3, shuffle=True, random_state=SEED) if groups is not None else StratifiedKFold(3, shuffle=True, random_state=SEED)


def evaluate(name, pipe, grid, X, y, groups, n_repeats):
    """Nested CV: outer folds estimate performance, inner folds tune the grid."""
    outer = (
        [(tr, te) for r in range(n_repeats)
         for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(X, y, groups)]
        if groups is not None
        else list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED).split(X, y))
    )
    def run_fold(tr, te):
        gs = GridSearchCV(pipe, grid, scoring="recall", cv=make_inner_cv(None if groups is None else groups[tr]), n_jobs=1)
        gs.fit(X.iloc[tr], y[tr], **({"groups": groups[tr]} if groups is not None else {}))
        return te, gs.predict_proba(X.iloc[te])[:, 1]

    fold_out = Parallel(n_jobs=-1)(delayed(run_fold)(tr, te) for tr, te in outer)
    rows, oof_sum, oof_n = [], np.zeros(len(y)), np.zeros(len(y))
    for te, proba in fold_out:
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        rows.append(dict(
            accuracy=accuracy_score(y[te], pred),
            recall=tp / max(tp + fn, 1),
            specificity=tn / max(tn + fp, 1),
            f1=f1_score(y[te], pred, zero_division=0),
            roc_auc=roc_auc_score(y[te], proba) if len(set(y[te])) > 1 else np.nan,
        ))
        oof_sum[te] += proba
        oof_n[te] += 1
    df = pd.DataFrame(rows)
    out = {"model": name, **{f"{m}_mean": df[m].mean() for m in df}, **{f"{m}_std": df[m].std() for m in df}}
    return out, oof_sum / np.maximum(oof_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV/XLSX with the 80 feature columns + label")
    ap.add_argument("--label", default="label")
    ap.add_argument("--group", default=None, help="column identifying the person, if rows repeat per person")
    ap.add_argument("--repeats", type=int, default=10, help="outer CV repeats (default 10 x 5 folds)")
    ap.add_argument("--acc-floor", type=float, default=0.70, help="minimum mean accuracy to be eligible")
    ap.add_argument("--target-recall", type=float, default=None,
                    help="optional: lower the decision threshold until out-of-fold recall reaches this")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of PEOPLE (e.g. 0.15) kept out of model selection and scored once at the end")
    ap.add_argument("--uncertain-width", type=float, default=0.10,
                    help="app shows 'inconclusive' when probability is within threshold +/- this")
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    path = Path(args.data)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing[:8]} ...")
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y = df[args.label].astype(int).to_numpy()
    groups = df[args.group].to_numpy() if args.group else None
    print(f"{len(X)} rows, {X.shape[1]} features, class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
    if groups is None:
        print("NOTE: no --group given -> assuming ONE row per person.")

    X_all, y_all, groups_all = X, y, groups
    hold = None
    if args.holdout > 0:
        sp = StratifiedGroupKFold(max(2, round(1 / args.holdout)), shuffle=True, random_state=SEED)
        dev_idx, hold_idx = next(sp.split(X, y, groups if groups is not None else np.arange(len(y))))
        hold = (X.iloc[hold_idx], y[hold_idx])
        X, y = X.iloc[dev_idx].reset_index(drop=True), y[dev_idx]
        groups = groups[dev_idx] if groups is not None else None
        print(f"Holdout: {len(hold_idx)} rows kept aside; model selection uses {len(dev_idx)} rows.")

    results, oofs = [], {}
    models = build_models()
    for name, (pipe, grid) in models.items():
        print(f"evaluating {name} ...", flush=True)
        res, oof = evaluate(name, pipe, grid, X, y, groups, args.repeats)
        results.append(res)
        oofs[name] = oof
    res = pd.DataFrame(results).sort_values("recall_mean", ascending=False)

    # Selection: highest recall among models that meet the accuracy floor; AUC breaks ties.
    # Recall alone is gameable (predict "Parkinson's" for everyone => recall 1.0), so we also
    # require accuracy and report specificity.
    eligible = res[res.accuracy_mean >= args.acc_floor]
    if eligible.empty:
        print(f"No model reached accuracy >= {args.acc_floor}; falling back to best ROC-AUC.")
        best_name = res.sort_values("roc_auc_mean", ascending=False).iloc[0]["model"]
    else:
        best_name = eligible.sort_values(["recall_mean", "roc_auc_mean"], ascending=False).iloc[0]["model"]
    res["selected"] = res["model"] == best_name

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out / "model_comparison.csv", index=False)
    cols = ["model", "accuracy_mean", "recall_mean", "specificity_mean", "f1_mean", "roc_auc_mean", "selected"]
    print("\n" + res[cols].round(3).to_string(index=False))

    # Optional threshold: highest threshold whose out-of-fold recall >= target.
    threshold = 0.5
    if args.target_recall:
        p = oofs[best_name]
        for t in np.arange(0.5, 0.0, -0.01):
            if ((p >= t) & (y == 1)).sum() / (y == 1).sum() >= args.target_recall:
                threshold = round(float(t), 2)
                break
        pred = (p >= threshold).astype(int)
        print(f"\nthreshold {threshold}: OOF recall={((pred == 1) & (y == 1)).sum() / (y == 1).sum():.3f}, "
              f"accuracy={accuracy_score(y, pred):.3f}")

    # Final fit: tune with CV, keep the refit best estimator.
    pipe, grid = models[best_name]
    final = GridSearchCV(pipe, grid, scoring="recall", cv=make_inner_cv(groups), n_jobs=1)
    final.fit(X, y, **({"groups": groups} if groups is not None else {}))

    holdout_metrics = None
    if hold is not None:  # scored exactly once, never used to choose anything
        hp = final.best_estimator_.predict_proba(hold[0])[:, 1]
        hpred = (hp >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(hold[1], hpred, labels=[0, 1]).ravel()
        holdout_metrics = dict(n=int(len(hold[1])), accuracy=round(float(accuracy_score(hold[1], hpred)), 4),
                               recall=round(float(tp / max(tp + fn, 1)), 4), specificity=round(float(tn / max(tn + fp, 1)), 4),
                               roc_auc=round(float(roc_auc_score(hold[1], hp)), 4) if len(set(hold[1])) > 1 else None)
        print(f"\nHOLDOUT (unseen people, scored once): {holdout_metrics}")
        # now use every row for the deployed model
        X, y, groups = X_all, y_all, groups_all
        final = GridSearchCV(pipe, grid, scoring="recall", cv=make_inner_cv(groups), n_jobs=1)
        final.fit(X, y, **({"groups": groups} if groups is not None else {}))

    reference = {  # raw-feature stats per class, used by the app to show which features look unusual
        "control_mean": X[y == 0].mean().to_dict(), "control_std": X[y == 0].std().replace(0, np.nan).to_dict(),
        "pd_mean": X[y == 1].mean().to_dict(),
    }
    bundle = {
        "model": final.best_estimator_,
        "feature_names": FEATURES,
        "threshold": threshold,
        "uncertain_band": [round(threshold - args.uncertain_width, 3), round(threshold + args.uncertain_width, 3)],
        "model_name": best_name,
        "best_params": final.best_params_,
        "holdout_metrics": holdout_metrics,
        "reference": reference,
        "cv_metrics": {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) and not isinstance(v, (bool, np.bool_)) else str(v)) for k, v in res[res.selected].iloc[0].items()},
    }
    joblib.dump(bundle, out / "notebook_model.joblib")
    print(f"\nSaved '{best_name}' ({final.best_params_}) -> {out / 'notebook_model.joblib'}")


if __name__ == "__main__":
    main()
