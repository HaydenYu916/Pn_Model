import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from datetime import datetime

def minmax_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def fifth_order_poly(x, c0, c1, c2, c3, c4, c5):
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5

def curvature(x, y):
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    return d2y / (1 + dy**2)**1.5

def fit_and_knee_analysis(data_path, output_dir=None):
    print(f"Reading data: {data_path}")
    df = pd.read_csv(data_path)
    # Normalization
    f1 = minmax_normalize(df['Pn'].values)
    g1 = minmax_normalize(df['CLED'].values)
    # Axes
    u = f1
    h = g1
    # Sort by u
    idx = np.argsort(u)
    u_sorted = u[idx]
    h_sorted = h[idx]
    # 5th order polynomial fit
    # print("5th order polynomial fitting...")
    # p0 = [h_sorted.min(), 0, 0, 0, 0, 0]
    # popt, _ = curve_fit(fifth_order_poly, u_sorted, h_sorted, p0=p0, maxfev=10000)
    # u_fit = np.linspace(u_sorted.min(), u_sorted.max(), 2000)
    # h_fit = fifth_order_poly(u_fit, *popt)
    # Curvature
    # curv = curvature(u_fit, h_fit)
    # 只保留主区间 [0.2, 1.0]
    # main_mask = (u_fit >= 0.2) & (u_fit <= 1.0)
    # u_main = u_fit[main_mask]
    # h_main = h_fit[main_mask]
    # curv_main = curv[main_mask]
    # 在主区间找曲率最大点（主峰）
    # knee_idx = np.argmax(curv_main)
    # knee_u = u_main[knee_idx]
    # knee_h = h_main[knee_idx]
    # knee_curv = curv_main[knee_idx]
    # Find closest original point
    # orig_idx = np.argmin((u - knee_u)**2 + (h - knee_h)**2)
    # orig_row = df.iloc[orig_idx]
    # Output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/pareto_knee_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    # 推荐点：Pn - CLED/max(CLED) 加权最大
    score = df['Pn'] - df['CLED'] / df['CLED'].max()
    best_idx = score.idxmax()
    best_row = df.loc[best_idx]
    fit_params = {
        'recommended_point': {
            'PPFD': float(best_row['PPFD']),
            'R:B': float(best_row['R:B']),
            'CLED': float(best_row['CLED']),
            'Pn': float(best_row['Pn'])
        }
    }
    import json
    with open(os.path.join(output_dir, 'fit_knee_parameters.json'), 'w', encoding='utf-8') as f:
        json.dump(fit_params, f, indent=2, ensure_ascii=False)
    print('\n===== Recommended Point =====')
    print(f'PPFD = {best_row["PPFD"]:.2f}, R:B = {best_row["R:B"]:.4f}, CLED = {best_row["CLED"]:.2f}, Pn = {best_row["Pn"]:.4f}')
    print(f'All results saved to: {output_dir}')
    return fit_params

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        fit_and_knee_analysis(data_path, output_dir)
    else:
        print("用法: python fit.py <pareto_solutions.csv 路径> [输出目录]")
