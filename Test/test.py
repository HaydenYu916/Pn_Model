import requests
import openai
from datetime import datetime

# 🔧 设置你的 OpenAI API Key
client = openai.OpenAI(api_key="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")

# 🚀 1. 获取 AEMO 电价数据
resp = requests.get("https://visualisations.aemo.com.au/aemo/apps/api/report/ELEC_NEM_SUMMARY")
data = resp.json()

# 🧮 2. 提取 NSW1 最新价格记录
elec = data["ELEC_NEM_SUMMARY"]
nsw = sorted([r for r in elec if r["REGIONID"] == "NSW1"], key=lambda x: x["SETTLEMENTDATE"])[-1]

# 🔍 3. 构建自然语言 Prompt
price = float(nsw["PRICE"])
demand = float(nsw["TOTALDEMAND"])
ts = nsw["SETTLEMENTDATE"]

prompt = f"""
你是一个智能电价分析助手。

以下是从 AEMO 获取的 NSW 最新电力市场数据：

- 时间：{ts}
- 电价（Spot Price）：{price:.2f} $/MWh
- 总需求：{demand:.1f} MW

请完成以下任务：
1. 判断当前电价是否高于 300 $/MWh？
2. 电价是否处于偏高、偏低、正常区间？（可参考：低于100为偏低，100-300为正常，高于300为偏高）
3. 我是否应该关闭高耗能设备（如热水器）？
4. 简要说明原因。
请用简洁的语言直接输出结论。
"""

# 🤖 4. 调用 OpenAI GPT 模型
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一个能源分析助手"},
        {"role": "user", "content": prompt}
    ]
)

# 🖨️ 5. 输出回答
print(f"当前NSW电价：{price:.2f} $/MWh")
print("📊 当前 NSW Spot Price 电价分析建议：\n")
print(response.choices[0].message.content)
