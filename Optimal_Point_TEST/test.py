import nemseer
import pandas as pd
from datetime import datetime, timedelta

# 本地缓存目录（可修改）
RAW_CACHE = "./nemseer_raw_cache"
PROCESSED_CACHE = "./nemseer_processed_cache"

def round_to_5min(dt):
    # 四舍五入到最近的5分钟
    discard = timedelta(minutes=dt.minute % 5,
                        seconds=dt.second,
                        microseconds=dt.microsecond)
    dt -= discard
    if discard >= timedelta(minutes=2.5):
        dt += timedelta(minutes=5)
    return dt

# 获取当前时间和预测窗口
now = datetime.utcnow()
run_start = round_to_5min(now - timedelta(minutes=10)).strftime("%Y/%m/%d %H:%M")
run_end   = round_to_5min(now).strftime("%Y/%m/%d %H:%M")
fcast_start = run_end
fcast_end = round_to_5min(now + timedelta(minutes=30)).strftime("%Y/%m/%d %H:%M")

# 调用 nemseer 获取 P5MIN（5分钟预测）数据
dfs = nemseer.compile_data(
    run_start=run_start,
    run_end=run_end,
    forecasted_start=fcast_start,
    forecasted_end=fcast_end,
    forecast_type="P5MIN",
    tables=["REGIONPRICE"],
    raw_cache=RAW_CACHE,
    processed_cache=PROCESSED_CACHE,
    data_format="df"
)

# 提取 NSW 区域的价格预测（NSW1）
df = dfs["REGIONPRICE"]
nsw = df[df["REGIONID"] == "NSW1"]

# 显示最新一次运行的预测时间和未来 5 分钟电价
latest_run = nsw["RUN_DATETIME"].max()
forecast_next = nsw[nsw["RUN_DATETIME"] == latest_run].sort_values("FORECAST_DATETIME").head(1)

print(f"⏱️ 预测运行时间：{latest_run}")
for _, row in forecast_next.iterrows():
    print(f"📈 {row['FORECAST_DATETIME']} - 预测价格：${row['REGIONPRICE']:.2f} / MWh")
