import pandas as pd
import matplotlib.pyplot as plt


y_true_sample = pd.read_csv("eda/true.csv")["trip_duration"].sample(n=40000, random_state=42)
y_pred_sample = pd.read_csv("eda/pred.csv")["trip_duration"].sample(n=40000, random_state=42)

plt.figure(figsize=(6, 4), dpi=300)
plt.scatter(y_true_sample, y_pred_sample, alpha=0.3, color='#1f77b4', edgecolors='none', s=15)

# (Perfect Prediction Line)
max_val = max(max(y_true_sample), max(y_pred_sample))
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit')

plt.title('NYC Taxi Trip Duration: Actual vs Predicted', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Duration (Seconds)', fontsize=12)
plt.ylabel('Predicted Duration (Seconds)', fontsize=12)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle=':', alpha=0.6)

plt.xlim(0, 1000)
plt.ylim(0, 1000)

plt.tight_layout()
plt.savefig('eda/evaluation_plot.png', dpi=300)
plt.show()
