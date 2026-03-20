import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import SUMMARY_DIR

# ── 1. LOAD DATA ─────────────────────────────────────────────
file_path = os.path.join(SUMMARY_DIR, "final_results_summary.csv")
if not os.path.exists(file_path):
    print("❌ File not found. Please run 4_run_shap_and_summary.py first.")
    exit()

df = pd.read_csv(file_path)

# ── 2. SELECT SCENARIOS WITH NOISELESS QUANTUM ADVANTAGE ─────
# Keep only scenarios where ΔAUC (Clean) > 0.03 (operational threshold)
advantage_scenarios = df[df['Delta_Clean'] > 0.03].copy()

if advantage_scenarios.empty:
    print("No scenarios with ΔAUC > 0.03 found.")
    exit()

# ── 3. RESHAPE TO LONG FORM FOR SEABORN ──────────────────────
plot_data = pd.melt(
    advantage_scenarios,
    id_vars=['Dataset', 'Features', 'Samples', 'Imb_Ratio'],
    value_vars=['AUC_XGB', 'AUC_QSVM_Clean', 'AUC_QSVM_Noisy'],
    var_name='Model', value_name='AUC Score'
)

plot_data['Scenario'] = (
    plot_data['Dataset'] + "\n" +
    "F=" + plot_data['Features'].astype(str) +
    " N=" + plot_data['Samples'].astype(str) + "\n" +
    plot_data['Imb_Ratio']
)

# Rename model labels for the legend
label_map = {
    'AUC_XGB':         'XGBoost',
    'AUC_QSVM_Clean':  'QSVM (Noiseless)',
    'AUC_QSVM_Noisy':  'QSVM (NISQ Noisy)'
}
plot_data['Model'] = plot_data['Model'].map(label_map)

# ── 4. PLOT ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
sns.set_style("whitegrid")

# Colour scheme suitable for print and colour-blind readers
colors = ["#7f8c8d", "#27ae60", "#e74c3c"]

sns.barplot(
    data=plot_data, x='Scenario', y='AUC Score',
    hue='Model', palette=colors, ax=ax,
    order=plot_data['Scenario'].unique()
)

# Add value labels on bars
for p in ax.patches:
    h = p.get_height()
    if pd.notna(h) and h > 0.01:
        ax.annotate(
            f"{h:.3f}",
            (p.get_x() + p.get_width() / 2.0, h),
            ha='center', va='bottom',
            xytext=(0, 4), textcoords='offset points',
            fontsize=8, fontweight='bold'
        )

ax.set_title(
    'Comparison of AUC Scores: Classical vs. Quantum (Noiseless and NISQ-Noisy Simulations)',
    fontsize=13, fontweight='bold', pad=14
)
ax.set_ylabel('AUC Score', fontsize=11)
ax.set_xlabel('Experimental Scenarios', fontsize=11)
ax.set_ylim(0.40, 1.05)
ax.legend(title='Models', loc='lower right', fontsize=10)

# Error bars note
fig.text(
    0.5, -0.02,
    'Note: Bars represent mean AUC across 50 cross-validation replications (noiseless) '
    'and 5 folds (noisy). ΔAUC = AUC_QSVM − AUC_XGB.',
    ha='center', fontsize=8, color='#555555'
)

plt.tight_layout()

# ── 5. SAVE ───────────────────────────────────────────────────
output_path = os.path.join(SUMMARY_DIR, "figure1_performance_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {output_path}")
plt.show()
