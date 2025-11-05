# tools/analyze_reports.py
# Универсальный анализатор отчётов из data/out/**/report.json
# Делает: plants.csv, defects.csv, графики, корреляции, регрессию, деревья, KNN, KMeans, summary.md

from __future__ import annotations
import argparse, json, math, os, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

plt.rcParams["figure.dpi"] = 120

def read_reports(in_dir: Path) -> list[dict]:
    reports = []
    for rp in in_dir.rglob("report.json"):
        try:
            reports.append(json.loads(rp.read_text(encoding="utf-8")))
        except Exception:
            pass
    return reports

def build_tables(reports: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    plants_rows, defects_rows = [], []
    for data in reports:
        rid = data.get("request_id")
        # plants
        for p in data.get("plants", []) or []:
            plants_rows.append({
                "request_id": rid,
                "plant_id": p.get("id"),
                "cls": p.get("cls"),
                "conf": p.get("conf"),
                "area": p.get("area"),
                "tilt_deg": p.get("tilt_deg"),
                "dry_ratio": p.get("dry_ratio"),
                "species": p.get("species"),
                "health_score": p.get("health_score"),
                "health_grade": p.get("health_grade"),
            })
        # defects
        for d in data.get("defects", []) or []:
            defects_rows.append({
                "request_id": rid,
                "defect_id": d.get("id"),
                "plant_id": d.get("plant_id"),
                "cls": d.get("cls"),
                "conf": d.get("conf"),
                "area": d.get("area"),
                "area_ratio": d.get("area_ratio"),
            })
    return pd.DataFrame(plants_rows), pd.DataFrame(defects_rows)

def ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)

def save_df(df: pd.DataFrame, path: Path):
    if not df.empty:
        df.to_csv(path, index=False, encoding="utf-8")

def plot_hist(series, title, path):
    series = pd.Series(series).dropna()
    if series.empty: return
    plt.figure()
    series.hist(bins=30)
    plt.title(title); plt.xlabel(title); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(path); plt.close()

def bar_top_counts(series, title, path, top=20):
    vc = series.value_counts().head(top)
    if vc.empty: return
    plt.figure()
    vc.plot(kind="bar")
    plt.title(title); plt.xlabel("class"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(path); plt.close()

def corr_heatmap(df, title, path):
    if df.shape[1] < 2:
        return
    C = df.corr(method="spearman")
    plt.figure(figsize=(6,5))
    im = plt.imshow(C, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(C.columns)), C.columns, rotation=45, ha="right")
    plt.yticks(range(len(C.index)), C.index)
    for (i,j), val in np.ndenumerate(C.values):
        plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout(); plt.savefig(path); plt.close()

def make_health_bins(s: pd.Series) -> pd.Series:
    s = s.fillna(s.median())
    bins = [-1,25,50,75,200]
    labels = ["critical","poor","fair","good"]
    return pd.cut(s, bins=bins, labels=labels)

def regression_block(X: pd.DataFrame, y: pd.Series, figs_dir: Path) -> dict:
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())
    if X.shape[0] < 40:
        return {"note": "too_few_samples_for_regression"}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    # Linear
    lr = Pipeline([("scaler", StandardScaler(with_mean=False)), ("lr", LinearRegression(n_jobs=None))])
    lr.fit(Xtr, ytr); pred_lr = lr.predict(Xte)
    # Ridge
    ridge = Pipeline([("scaler", StandardScaler(with_mean=False)), ("ridge", Ridge())])
    ridge.fit(Xtr, ytr); pred_rg = ridge.predict(Xte)
    def metrics(y_true, y_pred):
        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": math.sqrt(mean_squared_error(y_true, y_pred))
        }
    m_lr = metrics(yte, pred_lr); m_rg = metrics(yte, pred_rg)
    # residuals plot
    plt.figure()
    plt.scatter(pred_rg, yte - pred_rg, s=10)
    plt.axhline(0, ls="--", lw=1)
    plt.xlabel("Fitted (Ridge)"); plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout(); plt.savefig(figs_dir/"residuals_ridge.png"); plt.close()
    return {"linear": m_lr, "ridge": m_rg, "n_test": int(len(yte))}

def tree_knn_block(X: pd.DataFrame, y: pd.Series, figs_dir: Path) -> dict:
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.astype(str)
    if X.shape[0] < 40 or y.nunique() < 2:
        return {"note": "too_few_samples_for_classification"}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=None, random_state=42)
    tree.fit(Xtr, ytr)
    pred_t = tree.predict(Xte)
    rep_t = classification_report(yte, pred_t, digits=3, output_dict=True)
    cm_t = confusion_matrix(yte, pred_t, labels=sorted(y.unique()))
    plt.figure()
    plt.imshow(cm_t); plt.title("DecisionTree — Confusion")
    plt.xticks(range(len(cm_t)), sorted(y.unique()), rotation=45, ha="right")
    plt.yticks(range(len(cm_t)), sorted(y.unique()))
    plt.tight_layout(); plt.savefig(figs_dir/"tree_confusion.png"); plt.close()
    # KNN
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    gs = GridSearchCV(pipe, {"knn__n_neighbors":[3,5,7,9,11]}, scoring="f1_macro", cv=5, n_jobs=-1)
    gs.fit(Xtr, ytr)
    pred_k = gs.predict(Xte)
    rep_k = classification_report(yte, pred_k, digits=3, output_dict=True)
    return {
        "tree": {"f1_macro": rep_t.get("macro avg",{}).get("f1-score", None), "best_depth": tree.get_depth()},
        "knn": {"f1_macro": rep_k.get("macro avg",{}).get("f1-score", None), "best_k": gs.best_params_.get("knn__n_neighbors")},
    }

def kmeans_block(X: pd.DataFrame, figs_dir: Path) -> dict:
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if X.shape[0] < 20:
        return {"note": "too_few_samples_for_kmeans"}
    sc = StandardScaler()
    Xs = sc.fit_transform(X.values)
    ks = range(2, 8)
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(Xs)
        if Xs.shape[0] > k:
            scores.append(silhouette_score(Xs, km.labels_))
        else:
            scores.append(np.nan)
    best_k = ks[int(np.nanargmax(scores))] if np.isfinite(scores).any() else 3
    km = KMeans(n_clusters=best_k, n_init="auto", random_state=42).fit(Xs)
    # plot inertia / silhouette
    plt.figure(); plt.plot(list(ks), [KMeans(n_clusters=k, n_init="auto", random_state=42).fit(Xs).inertia_ for k in ks], marker="o")
    plt.title("KMeans — Elbow (inertia)"); plt.xlabel("k")
    plt.tight_layout(); plt.savefig(figs_dir/"kmeans_elbow.png"); plt.close()
    plt.figure(); plt.plot(list(ks), scores, marker="o"); plt.title("KMeans — Silhouette")
    plt.tight_layout(); plt.savefig(figs_dir/"kmeans_silhouette.png"); plt.close()
    # centroids
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=X.columns)
    cent["cluster"] = range(best_k)
    cent.to_csv(figs_dir.parent/"kmeans_centroids.csv", index=False, encoding="utf-8")
    return {"best_k": int(best_k), "silhouette_best": float(np.nanmax(scores)) if np.isfinite(scores).any() else None}

def copy_overlay_examples(reports: list[dict], out_dir: Path, limit=3):
    saved = 0
    for d in reports:
        p = d.get("overlay_path")
        if not p: continue
        src = Path(p)
        # если путь контейнерный (/data/...), пробуем относительный аналог
        if not src.exists():
            alt = Path("."+str(src)).resolve()
            if alt.exists(): src = alt
        if src.exists():
            dst = out_dir / f"example_{saved+1}.png"
            try:
                shutil.copyfile(src, dst)
                saved += 1
            except Exception:
                pass
        if saved >= limit: break
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="data/out", help="директория с <rid>/report.json")
    ap.add_argument("--out", dest="out_dir", default="reports", help="куда складывать отчёты/графики")
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir); figs = out_dir / "figs"
    ensure_out(out_dir)

    reports = read_reports(in_dir)
    if not reports:
        print(f"[!] report.json не найден в {in_dir.resolve()}")
        return

    dfp, dfd = build_tables(reports)
    save_df(dfp, out_dir/"plants.csv")
    save_df(dfd, out_dir/"defects.csv")

    # 3. визуализации
    plot_hist(dfp.get("area"), "plant_area", figs/"plant_area_hist.png")
    plot_hist(dfd.get("area"), "defect_area", figs/"defect_area_hist.png")
    plot_hist(dfp.get("tilt_deg"), "tilt_deg", figs/"tilt_deg_hist.png")
    plot_hist(dfp.get("dry_ratio"), "dry_ratio", figs/"dry_ratio_hist.png")
    bar_top_counts(dfp.get("species").dropna(), "species balance", figs/"species_balance.png")
    bar_top_counts(dfd.get("cls").dropna(), "defects balance", figs/"defects_balance.png")
    overlays = copy_overlay_examples(reports, figs)

    # агрегаты дефектов на растение
    pivot = pd.DataFrame()
    if not dfd.empty:
        pivot = dfd.pivot_table(index=["request_id","plant_id"], columns="cls", values="area_ratio", aggfunc="sum").fillna(0)
        pivot.columns = [f"def_{c}" for c in pivot.columns]
    base = dfp.set_index(["request_id","plant_id"])[["area","tilt_deg","dry_ratio","health_score"]].copy()
    X_all = base.join(pivot, how="left").fillna(0)

    # 4. корреляции
    corr_cols = [c for c in X_all.columns if c not in []]
    corr_heatmap(X_all[corr_cols], "Spearman correlations", figs/"corr_heatmap.png")

    # 5. регрессия (target=health_score)
    y = X_all["health_score"]
    X_reg = X_all.drop(columns=["health_score"])
    reg_res = regression_block(X_reg, y, figs)

    # 6–7. классы по health_score бинами
    y_cls = make_health_bins(y)
    X_cls = X_reg.copy()
    clf_res = tree_knn_block(X_cls, y_cls, figs)

    # 8. KMeans
    km_res = kmeans_block(X_reg, figs)

    # summary
    md = []
    md += ["# Аналитический отчёт", ""]
    md += ["## Обзор входных данных", f"- plants: {len(dfp)} записей", f"- defects: {len(dfd)} записей", f"- примеры оверлеев: {overlays}"]
    md += ["", "## Визуализации", "- см. папку `reports/figs/`"]
    md += ["", "## Корреляции", "- `figs/corr_heatmap.png`"]
    md += ["", "## Регрессия (health_score)"]
    md += [f"- Linear: R2={reg_res.get('linear',{}).get('R2'):.3f}  MAE={reg_res.get('linear',{}).get('MAE'):.3f}  RMSE={reg_res.get('linear',{}).get('RMSE'):.3f}" if 'linear' in reg_res else "- недостаточно данных"]
    md += [f"- Ridge:  R2={reg_res.get('ridge',{}).get('R2'):.3f}  MAE={reg_res.get('ridge',{}).get('MAE'):.3f}  RMSE={reg_res.get('ridge',{}).get('RMSE'):.3f}" if 'ridge' in reg_res else ""]
    md += ["", "## Классификация (категории состояния)"]
    if "tree" in clf_res:
        md += [f"- DecisionTree: F1-macro={clf_res['tree']['f1_macro']:.3f}, depth={clf_res['tree']['best_depth']}"]
    if "knn" in clf_res:
        md += [f"- KNN: F1-macro={clf_res['knn']['f1_macro']:.3f}, best_k={clf_res['knn']['best_k']}"]
    md += ["", "## Кластеризация (KMeans)"]
    if "best_k" in km_res:
        md += [f"- best_k={km_res['best_k']}, silhouette={km_res.get('silhouette_best')}"]
        md += ["- центроиды: `reports/kmeans_centroids.csv`"]
    (out_dir/"summary.md").write_text("\n".join([s for s in md if s is not None]), encoding="utf-8")
    print(f"[OK] Готово. Отчёт в {out_dir.resolve()}")

if __name__ == "__main__":
    main()