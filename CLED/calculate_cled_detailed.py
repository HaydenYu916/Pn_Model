#!/usr/bin/env python3
"""
🌱 详细CLED计算脚本（精简版）：基于实际LED驱动器规格
仅批量计算CLED并保存结果，风格与calculate_cled.py一致。
"""
import pandas as pd
from typing import Tuple, Dict

class DetailedCLEDCalculator:
    def __init__(self):
        self.Ca = 581.0  # kg CO₂/MWh
        self.S = 1.0     # m²
        self.conversion_factor = 3600
        self.constant_current = 1.050  # A
        self.max_power_per_driver = 75.0  # W
        self.red_voltage = 35.0
        self.blue_voltage = 45.0
        self.boards_per_channel = 2
        self.red_voltage_total = self.red_voltage * self.boards_per_channel
        self.blue_voltage_total = self.blue_voltage * self.boards_per_channel
        self.red_power_per_channel = self.red_voltage_total * self.constant_current
        self.blue_power_per_channel = self.blue_voltage_total * self.constant_current
        self.red_efficiency = 2.8   # μmol/J
        self.blue_efficiency = 2.4  # μmol/J
        self.driver_efficiency = 0.92
        self.thermal_efficiency = 0.95
        self.optical_efficiency = 0.90
        self.system_efficiency = self.driver_efficiency * self.thermal_efficiency * self.optical_efficiency

    def decompose_light(self, ppfd_total: float, rb_ratio: float) -> Tuple[float, float]:
        red_ppfd = ppfd_total * rb_ratio
        blue_ppfd = ppfd_total * (1 - rb_ratio)
        return red_ppfd, blue_ppfd

    def calculate_power_consumption(self, ppfd_red: float, ppfd_blue: float) -> float:
        red_energy_density = ppfd_red / self.red_efficiency
        blue_energy_density = ppfd_blue / self.blue_efficiency
        red_actual_power = red_energy_density / self.system_efficiency
        blue_actual_power = blue_energy_density / self.system_efficiency
        total_power = red_actual_power + blue_actual_power
        return total_power

    def calculate_cled(self, ppfd: float, rb_ratio: float) -> float:
        if ppfd <= 0:
            return 0.0
        red_ppfd, blue_ppfd = self.decompose_light(ppfd, rb_ratio)
        total_power = self.calculate_power_consumption(red_ppfd, blue_ppfd)
        Ca_g_per_kwh = self.Ca * 0.001
        total_carbon_emission = (total_power * Ca_g_per_kwh) / 1000
        cled = total_carbon_emission * 1000 / self.conversion_factor
        return cled

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔄 开始批量计算详细CLED...")
        df['CLED_Detailed'] = df.apply(
            lambda row: self.calculate_cled(row['PPFD'], row['R:B']), axis=1)
        print(f"✅ 完成！处理了 {len(df)} 个数据点")
        return df

def main():
    print("🌱 详细LED碳排放计算 (精简版)")
    print("=" * 60)
    try:
        df = pd.read_csv('averaged_data.csv')
        print(f"✅ 成功加载数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    if not all(col in df.columns for col in ['PPFD', 'R:B']):
        print("❌ 缺少必要列: PPFD, R:B")
        return
    calculator = DetailedCLEDCalculator()
    df_with_cled = calculator.process_dataframe(df.copy())
    print("\n📊 详细CLED统计:")
    stats = df_with_cled['CLED_Detailed'].describe()
    print(stats.round(4))
    output_file = 'averaged_data_with_cled_detailed.csv'
    df_with_cled.to_csv(output_file, index=False)
    print(f"✅ 结果已保存: {output_file}")

if __name__ == "__main__":
    main() 