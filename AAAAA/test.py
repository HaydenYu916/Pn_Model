import requests
import json
from datetime import datetime

def get_nsw_price():
    url = "https://api.openelectricity.org.au/v4/data/network/NEM"
    params = {
        "metrics": "price",
        "interval": "5m",
        "network_region": "NSW1"
    }
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json'
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        print("状态码：", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            print("API响应结构：", json.dumps(data, indent=2)[:500])
            if "data" in data and len(data["data"]) > 0:
                price_series = data["data"][0]
                if "history" in price_series and "data" in price_series["history"]:
                    latest = price_series["history"]["data"][-1]
                    price = latest[1]  # $/MWh
                    price_kwh = price / 1000
                    label = "低" if price_kwh < 0.05 else "中" if price_kwh < 0.15 else "高"
                    interval = price_series["history"].get("last", "未知时间")
                    return round(price_kwh, 4), label, interval
                else:
                    print("未找到history字段")
            else:
                print("未找到data字段或数据为空")
        else:
            print("响应内容：", resp.text[:200])
    except Exception as e:
        print("请求异常：", e)
    return None, None, None

print("正在获取NSW电价数据...")
price, label, interval = get_nsw_price()
if price is not None:
    print(f"当前 NSW 电价：{price} $/kWh，等级：{label}，时间：{interval}")
else:
    print("无法获取电价信息，请检查API文档或联系服务方。")
