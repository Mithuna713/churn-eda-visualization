"""
Case Study: Exploratory Data Analysis and Visualization of a Real-World Dataset
Course  : U21ADP05 - Exploratory Data Analysis and Visualization
Dataset : IBM Telco Customer Churn (7,043 customers x 21 fields)
Task    : Predict whether a telecom customer will churn (binary classification)
Model   : Multi-Layer Perceptron (Keras / TensorFlow)

Pipeline follows the 7 stages of data visualization (Ben Fry):
  1 Acquire -> 2 Parse -> 3 Filter -> 4 Mine -> 5 Represent -> 6 Refine -> 7 Interact
"""
import os, json, random, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, average_precision_score)
import tensorflow as tf
from tensorflow import keras
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SEED = 42
random.seed(SEED); np.random.seed(SEED); keras.utils.set_random_seed(SEED)
FIG_DIR = "figs"; os.makedirs(FIG_DIR, exist_ok=True)
RESULTS = {}

# ----------------------------------------------------------------------------
# Plot style: Times-style serif font, black text, colour palette
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.titlecolor": "black",
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5,
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
})
C_NO, C_YES = "#5B9BD5", "#D9534F"      # blue = stayed, red = churned
PAL = {"No": C_NO, "Yes": C_YES}

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name)); plt.close()

# ============================================================================
# STAGE 1 - ACQUIRE : load the raw dataset
# ============================================================================
URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
df = pd.read_csv("telco.csv") if os.path.exists("telco.csv") else pd.read_csv(URL)
RESULTS["shape_raw"] = list(df.shape)
RESULTS["dtypes_raw"] = {c: str(t) for c, t in df.dtypes.items()}

# ============================================================================
# STAGE 2 - PARSE : fix data types
# ============================================================================
# TotalCharges is stored as text because 11 rows contain a blank string
blank_total = (df["TotalCharges"].astype(str).str.strip() == "").sum()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
RESULTS["blank_totalcharges"] = int(blank_total)

# ============================================================================
# STAGE 3 - FILTER : missing values, duplicates, outliers, cleaning
# ============================================================================
RESULTS["missing_after_parse"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
# The blank rows are customers with tenure = 0 (joined this month, nothing billed yet)
RESULTS["blank_rows_tenure"] = df.loc[df["TotalCharges"].isna(), "tenure"].tolist()
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

RESULTS["duplicate_rows_full"] = int(df.duplicated().sum())
RESULTS["customerid_unique"] = bool(df["customerID"].is_unique)
RESULTS["duplicate_rows_excluding_id"] = int(df.drop(columns="customerID").duplicated().sum())

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
outliers = {}
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers[c] = {"lower": float(lo), "upper": float(hi),
                   "count": int(((df[c] < lo) | (df[c] > hi)).sum())}
RESULTS["outliers_iqr"] = outliers

# Cleaning: collapse redundant levels (information already held by other columns)
for c in ["MultipleLines"]:
    df[c] = df[c].replace("No phone service", "No")
addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
for c in addon_cols:
    df[c] = df[c].replace("No internet service", "No")
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

# ============================================================================
# STAGE 4 - MINE : descriptive statistics and hypothesis tests
# ============================================================================
RESULTS["describe"] = df[num_cols].describe().round(2).to_dict()
RESULTS["skew"] = {c: round(float(df[c].skew()), 3) for c in num_cols}
RESULTS["kurtosis"] = {c: round(float(df[c].kurt()), 3) for c in num_cols}
RESULTS["churn_rate"] = round(float(df["Churn_bin"].mean()), 4)
RESULTS["churn_counts"] = df["Churn"].value_counts().to_dict()
RESULTS["group_means"] = df.groupby("Churn")[num_cols].mean().round(2).to_dict()
RESULTS["group_medians"] = df.groupby("Churn")[num_cols].median().round(2).to_dict()

cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in ["customerID", "Churn", "tenure_band"]]
chi_rows = []
for c in cat_cols:
    ct = pd.crosstab(df[c], df["Churn"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    chi_rows.append({"feature": c, "chi2": round(float(chi2), 1), "dof": int(dof),
                     "p": float(p), "cramers_v": round(float(v), 3)})
chi_rows.sort(key=lambda r: -r["cramers_v"])
RESULTS["chi_square"] = chi_rows

mw = {}
for c in num_cols:
    a = df.loc[df.Churn == "Yes", c]; b = df.loc[df.Churn == "No", c]
    u, p = stats.mannwhitneyu(a, b)
    mw[c] = {"U": float(u), "p": float(p)}
RESULTS["mann_whitney"] = mw
RESULTS["pointbiserial"] = {c: round(float(df[c].corr(df["Churn_bin"])), 3) for c in num_cols}

churn_by = {}
for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen",
          "PaperlessBilling", "Partner", "Dependents", "OnlineSecurity"]:
    churn_by[c] = (df.groupby(c)["Churn_bin"].mean() * 100).round(1).to_dict()
RESULTS["churn_rate_by"] = churn_by

# tenure bands
df["tenure_band"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
RESULTS["churn_by_tenure_band"] = (df.groupby("tenure_band")["Churn_bin"].mean() * 100).round(1).to_dict()

# ============================================================================
# STAGE 5 & 6 - REPRESENT + REFINE : visualisations (colour, labelled)
# ============================================================================
# --- Fig 1: class balance --------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.6))
vc = df["Churn"].value_counts().reindex(["No", "Yes"])
bars = ax.bar(["Stayed (No)", "Churned (Yes)"], vc.values, color=[C_NO, C_YES], edgecolor="black")
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=10)
ax.set_ylabel("Number of customers"); ax.set_title("Target Class Distribution (Churn)")
ax.set_ylim(0, vc.max() * 1.18); ax.grid(axis="x", visible=False)
save("fig01_class_distribution.png")

# --- Fig 2: univariate histograms ------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
for ax, c, u in zip(axs, num_cols, ["months", "INR/USD per month", "total billed"]):
    sns.histplot(df[c], bins=30, kde=True, color="#4C78A8", edgecolor="black", ax=ax, line_kws={"color": "#1F3A5F", "lw": 1.8})
    ax.axvline(df[c].mean(), color="#D9534F", ls="--", lw=1.6, label=f"Mean = {df[c].mean():.1f}")
    ax.axvline(df[c].median(), color="#2E8B57", ls="-.", lw=1.6, label=f"Median = {df[c].median():.1f}")
    ax.set_title(f"Distribution of {c}"); ax.set_xlabel(c); ax.set_ylabel("Frequency"); ax.legend()
save("fig02_numeric_histograms.png")

# --- Fig 3: boxplots by churn ----------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, c in zip(axs, num_cols):
    sns.boxplot(data=df, x="Churn", y=c, order=["No", "Yes"], palette=PAL, linecolor="black",
                linewidth=1.1, ax=ax, flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "black"})
    ax.set_title(f"{c} by Churn Status"); ax.set_xlabel("Churn"); ax.set_ylabel(c)
save("fig03_boxplots_by_churn.png")

# --- Fig 4: churn rate by categorical features -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for ax, c in zip(axs.ravel(), ["Contract", "InternetService", "PaymentMethod", "TechSupport"]):
    rate = (df.groupby(c)["Churn_bin"].mean() * 100).sort_values(ascending=False)
    ax.barh(rate.index, rate.values, color="#4C78A8", edgecolor="black")
    for i, v in enumerate(rate.values):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.axvline(df["Churn_bin"].mean() * 100, color="#D9534F", ls="--", lw=1.5)
    ax.set_xlim(0, rate.max() * 1.18); ax.invert_yaxis()
    ax.set_title(f"Churn Rate by {c}"); ax.set_xlabel("Churn rate (%)"); ax.grid(axis="y", visible=False)
axs[0, 0].text(df["Churn_bin"].mean() * 100 + 0.5, -0.45, "dashed line = overall churn rate", fontsize=8)
save("fig04_churn_rate_by_category.png")

# --- Encoding (needed for correlation heatmap and model) -------------------
X = df.drop(columns=["customerID", "Churn", "Churn_bin", "tenure_band"]).copy()
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]) and set(X[c].unique()) <= set(binary_map):
        X[c] = X[c].map(binary_map)
X["n_addons"] = (df[addon_cols] == "Yes").sum(axis=1)       # engineered feature
multi = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
X = pd.get_dummies(X, columns=multi, drop_first=True, dtype=int)
y = df["Churn_bin"].values
RESULTS["n_features_encoded"] = int(X.shape[1])
RESULTS["feature_names"] = list(X.columns)

# --- Fig 5: correlation heatmap --------------------------------------------
sel = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Partner", "Dependents",
       "PhoneService", "PaperlessBilling", "n_addons", "Contract_One year", "Contract_Two year",
       "InternetService_Fiber optic", "PaymentMethod_Electronic check"]
cm = X[sel].assign(Churn=y).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, linewidths=.5, linecolor="white",
            annot_kws={"size": 8}, cbar_kws={"label": "Pearson correlation"}, ax=ax)
ax.set_title("Correlation Heatmap of Key Features and Churn"); ax.grid(False)
plt.xticks(rotation=45, ha="right")
save("fig05_correlation_heatmap.png")
RESULTS["corr_with_churn"] = cm["Churn"].drop("Churn").round(3).sort_values().to_dict()

# --- Fig 6: scatter ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.8))
for lab, m, col, a in [("No", "o", C_NO, .55), ("Yes", "x", C_YES, .75)]:
    s = df[df.Churn == lab]
    ax.scatter(s["tenure"], s["MonthlyCharges"], marker=m, c=col, alpha=a, s=18,
               edgecolors="black" if m == "o" else None, linewidths=.3,
               label=f"Churn = {lab}")
ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Monthly charges")
ax.set_title("Monthly Charges vs Tenure, by Churn Status"); ax.legend()
save("fig06_scatter_tenure_monthly.png")

# --- Fig 7: discrete distribution (add-on services ~ Binomial?) ------------
net = df[df["InternetService"] != "No"]
k = (net[addon_cols] == "Yes").sum(axis=1)
n = len(addon_cols); p_hat = k.mean() / n
obs = k.value_counts(normalize=True).reindex(range(n + 1), fill_value=0)
exp = stats.binom.pmf(range(n + 1), n, p_hat)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.bar(range(n + 1), obs.values, color="#7FB3D5", edgecolor="black", label="Observed")
ax.plot(range(n + 1), exp, "o-", color="#C0392B", lw=2, label=f"Binomial(n=6, p={p_hat:.2f})")
ax.set_xlabel("Number of add-on services subscribed"); ax.set_ylabel("Proportion of internet customers")
ax.set_title("Add-on Service Count vs Fitted Binomial Distribution"); ax.legend()
save("fig07_discrete_binomial.png")
chi_stat = float((len(k) * (obs.values - exp) ** 2 / exp).sum())
RESULTS["binomial_fit"] = {"n_internet_customers": int(len(k)), "p_hat": round(float(p_hat), 3),
    "mean": round(float(k.mean()), 3), "var_observed": round(float(k.var()), 3),
    "var_binomial": round(float(n * p_hat * (1 - p_hat)), 3), "chi2_gof": round(chi_stat, 1),
    "observed": obs.round(3).to_dict(), "expected": dict(zip(range(n + 1), np.round(exp, 3)))}

# --- Fig 8: continuous distribution fits + Q-Q -----------------------------
tc = df["TotalCharges"]; tcp = tc[tc > 0]
mu, sd = stats.norm.fit(tc); sh, loc, sc = stats.lognorm.fit(tcp, floc=0); lam = stats.expon.fit(tc, floc=0)[1]
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
xs = np.linspace(0, tc.max(), 400)
axs[0].hist(tc, bins=40, density=True, color="#A9CCE3", edgecolor="black", label="Observed")
axs[0].plot(xs, stats.norm.pdf(xs, mu, sd), "--", color="#C0392B", lw=2, label="Normal fit")
axs[0].plot(xs, stats.expon.pdf(xs, 0, lam), "-", color="#27AE60", lw=2, label="Exponential fit")
axs[0].plot(xs[1:], stats.lognorm.pdf(xs[1:], sh, loc, sc), ":", color="#8E44AD", lw=2.4, label="Log-normal fit")
axs[0].set_xlabel("TotalCharges"); axs[0].set_ylabel("Density")
axs[0].set_title("TotalCharges with Fitted Continuous Distributions"); axs[0].legend()
(osm, osr), (slope, icpt, r) = stats.probplot(tc, dist="norm")
axs[1].scatter(osm, osr, s=8, c="#2E86C1"); axs[1].plot(osm, slope * np.array(osm) + icpt, "-", color="#C0392B", lw=1.8)
axs[1].set_xlabel("Theoretical quantiles (Normal)"); axs[1].set_ylabel("Sample quantiles")
axs[1].set_title("Q-Q Plot of TotalCharges")
save("fig08_continuous_fits_qq.png")
RESULTS["continuous_fit"] = {
    "KS_normal": round(float(stats.kstest(tc, "norm", args=(mu, sd)).statistic), 3),
    "KS_exponential": round(float(stats.kstest(tc, "expon", args=(0, lam)).statistic), 3),
    "KS_lognormal": round(float(stats.kstest(tcp, "lognorm", args=(sh, loc, sc)).statistic), 3),
    "normal_mu_sd": [round(mu, 1), round(sd, 1)], "exp_mean": round(lam, 1),
    "qq_r": round(float(r), 3)}
mc = df["MonthlyCharges"]
RESULTS["monthly_shapiro_sample"] = round(float(stats.shapiro(mc.sample(2000, random_state=SEED)).statistic), 3)

# --- Fig 9: pairplot -----------------------------------------------------------
samp = df.sample(1200, random_state=SEED)
g = sns.pairplot(samp, vars=num_cols, hue="Churn", palette=PAL, diag_kind="kde",
                 plot_kws={"alpha": .6, "s": 14, "edgecolor": "black", "linewidth": .3},
                 corner=False, height=2.2)
g.fig.suptitle("Pairplot of Numerical Features (sample of 1,200 customers)", y=1.02, fontsize=12)
g.savefig(os.path.join(FIG_DIR, "fig09_pairplot.png")); plt.close("all")

# ============================================================================
# STAGE 7 - INTERACT : interactive HTML chart (hover, zoom, legend filtering)
# ============================================================================
try:
    import plotly.express as px
    fig_i = px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn", symbol="Contract",
                       hover_data=["InternetService", "PaymentMethod", "TotalCharges"],
                       title="Interactive: Monthly Charges vs Tenure")
    fig_i.write_html("interactive_tenure_vs_charges.html")
    RESULTS["interactive_html"] = True
except Exception as e:
    RESULTS["interactive_html"] = False

# ============================================================================
# PREPROCESSING : split 70/15/15, standardise numeric columns (train stats only)
# ============================================================================
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
scale_cols = ["tenure", "MonthlyCharges", "TotalCharges", "n_addons"]
sc_ = StandardScaler().fit(X_tr[scale_cols])
for part in (X_tr, X_va, X_te):
    part[scale_cols] = sc_.transform(part[scale_cols])
RESULTS["split"] = {"train": len(X_tr), "val": len(X_va), "test": len(X_te),
                    "train_churn": round(float(y_tr.mean()), 4), "val_churn": round(float(y_va.mean()), 4),
                    "test_churn": round(float(y_te.mean()), 4)}
Xtr, Xva, Xte = X_tr.values.astype("float32"), X_va.values.astype("float32"), X_te.values.astype("float32")

# ============================================================================
# DEEP LEARNING MODEL : MLP with basic hyper-parameter grid search
# ============================================================================
cw_neg = len(y_tr) / (2 * (y_tr == 0).sum()); cw_pos = len(y_tr) / (2 * (y_tr == 1).sum())
class_weight = {0: float(cw_neg), 1: float(cw_pos)}
RESULTS["class_weight"] = {k_: round(v, 3) for k_, v in class_weight.items()}

def build_mlp(n_in, units, dropout, lr):
    m = keras.Sequential([
        keras.layers.Input(shape=(n_in,)),
        keras.layers.Dense(units[0], activation="relu"), keras.layers.Dropout(dropout),
        keras.layers.Dense(units[1], activation="relu"), keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return m

grid = list(itertools.product([(32, 16), (64, 32), (128, 64)], [0.2, 0.4], [1e-3, 5e-3]))
best = {"auc": -1}; hp_rows = []
for units, dr, lr in grid:
    keras.utils.set_random_seed(SEED)
    m = build_mlp(Xtr.shape[1], units, dr, lr)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    h = m.fit(Xtr, y_tr, validation_data=(Xva, y_va), epochs=150, batch_size=64,
              class_weight=class_weight, callbacks=[es], verbose=0)
    va_auc = roc_auc_score(y_va, m.predict(Xva, verbose=0).ravel())
    hp_rows.append({"units": f"{units[0]}-{units[1]}", "dropout": dr, "lr": lr,
                    "epochs": len(h.history["loss"]), "val_auc": round(float(va_auc), 4)})
    if va_auc > best["auc"]:
        best = {"auc": va_auc, "model": m, "hist": h.history, "cfg": (units, dr, lr)}
RESULTS["hparam_grid"] = hp_rows
model, hist = best["model"], best["hist"]
RESULTS["best_cfg"] = {"units": list(best["cfg"][0]), "dropout": best["cfg"][1], "lr": best["cfg"][2],
                       "val_auc": round(float(best["auc"]), 4), "epochs_run": len(hist["loss"]),
                       "params": int(model.count_params())}

# ============================================================================
# EVALUATION
# ============================================================================
p_te = model.predict(Xte, verbose=0).ravel(); yhat = (p_te >= 0.5).astype(int)
RESULTS["test_metrics"] = {
    "accuracy": round(accuracy_score(y_te, yhat), 4), "precision": round(precision_score(y_te, yhat), 4),
    "recall": round(recall_score(y_te, yhat), 4), "f1": round(f1_score(y_te, yhat), 4),
    "auc": round(roc_auc_score(y_te, p_te), 4), "avg_precision": round(average_precision_score(y_te, p_te), 4)}
cmx = confusion_matrix(y_te, yhat); RESULTS["confusion_matrix"] = cmx.tolist()
lr_model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y_tr)
p_lr = lr_model.predict_proba(Xte)[:, 1]
RESULTS["baseline_logreg"] = {"auc": round(roc_auc_score(y_te, p_lr), 4),
    "accuracy": round(accuracy_score(y_te, (p_lr >= .5).astype(int)), 4),
    "recall": round(recall_score(y_te, (p_lr >= .5).astype(int)), 4)}
RESULTS["train_final"] = {"loss": round(hist["loss"][-1], 4), "val_loss": round(hist["val_loss"][-1], 4),
                          "acc": round(hist["accuracy"][-1], 4), "val_acc": round(hist["val_accuracy"][-1], 4)}
RESULTS["best_epoch"] = int(np.argmin(hist["val_loss"]) + 1)

ep = np.arange(1, len(hist["loss"]) + 1)
# Fig 10: loss
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(ep, hist["loss"], "-", color="#2E86C1", lw=2, label="Training loss")
ax.plot(ep, hist["val_loss"], "--", color="#E67E22", lw=2, label="Validation loss")
ax.axvline(RESULTS["best_epoch"], color="#7F8C8D", ls=":", lw=1.5, label=f"Best epoch = {RESULTS['best_epoch']}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy loss"); ax.set_title("Loss vs Epoch"); ax.legend()
save("fig10_loss_vs_epoch.png")
# Fig 11: accuracy
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(ep, hist["accuracy"], "-", color="#2E86C1", lw=2, label="Training accuracy")
ax.plot(ep, hist["val_accuracy"], "--", color="#E67E22", lw=2, label="Validation accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy vs Epoch"); ax.legend()
save("fig11_accuracy_vs_epoch.png")
# Fig 12: confusion matrix
fig, ax = plt.subplots(figsize=(4.8, 4.2))
sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor="white",
            annot_kws={"size": 14}, xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"], ax=ax)
ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label"); ax.set_title("Confusion Matrix (Test Set)"); ax.grid(False)
save("fig12_confusion_matrix.png")
# Fig 13: ROC
fpr, tpr, _ = roc_curve(y_te, p_te); fpr2, tpr2, _ = roc_curve(y_te, p_lr)
fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.plot(fpr, tpr, "-", color="#2E86C1", lw=2.2, label=f"MLP (AUC = {RESULTS['test_metrics']['auc']:.3f})")
ax.plot(fpr2, tpr2, "-.", color="#E67E22", lw=1.8, label=f"Logistic regression (AUC = {RESULTS['baseline_logreg']['auc']:.3f})")
ax.plot([0, 1], [0, 1], color="0.5", ls=":", label="Random classifier")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC Curve (Test Set)")
ax.legend(loc="lower right")
save("fig13_roc_curve.png")
# Fig 14: predicted probability distribution
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.hist(p_te[y_te == 0], bins=25, alpha=.9, color=C_NO, edgecolor="black", label="Actually stayed")
ax.hist(p_te[y_te == 1], bins=25, alpha=.75, color=C_YES, edgecolor="black", label="Actually churned")
ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Decision threshold = 0.5")
ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Number of customers")
ax.set_title("Distribution of Predicted Probabilities (Test Set)"); ax.legend()
save("fig14_probability_distribution.png")

# Permutation feature importance (drop in test AUC when a feature is shuffled)
rng = np.random.default_rng(SEED); base = roc_auc_score(y_te, p_te); imp = {}
for j, name in enumerate(X.columns):
    drops = []
    for _ in range(5):
        Xp = Xte.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(base - roc_auc_score(y_te, model.predict(Xp, verbose=0).ravel()))
    imp[name] = float(np.mean(drops))
top = sorted(imp.items(), key=lambda t: -t[1])[:10]
RESULTS["perm_importance_top10"] = [(k_, round(v, 4)) for k_, v in top]
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color="#4C78A8", edgecolor="black")
ax.set_xlabel("Decrease in test AUC after permuting feature"); ax.set_title("Top 10 Features by Permutation Importance")
ax.grid(axis="y", visible=False)
save("fig15_permutation_importance.png")

with open("results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print("Done. Figures:", sorted(os.listdir(FIG_DIR)))
