#!/usr/bin/env python3
"""
🌱 详细CLED计算脚本：基于实际LED驱动器规格
Detailed CLED Calculator Based on Actual LED Driver Specifications

基于提供的LED驱动器参数：
- Constant Current: 1050 mA
- Voltage Range: 36–71 V DC  
- Power Output: 75 W
- Red LED: ~35 V
- Blue LED: ~45 V
- 支持每通道串联2块板

公式：C_LED = (P_total × S × Ca) / (Eff_system × 3.6 × 10³)
输出单位：mg·m⁻²·s⁻¹
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
from datetime import datetime

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DetailedCLEDCalculator:
    """
    基于实际LED驱动器规格的详细CLED计算器
    
    使用具体的电压、电流参数计算精确的功率消耗和碳排放
    """
    
    def __init__(self):
        """初始化详细CLED计算参数"""
        # 🔧 论文环境参数
        self.Ca = 581.0  # 碳排因子 (kg CO₂/MWh)
        self.S = 1.0     # 照射面积 (m²)
        self.conversion_factor = 3600  # 转换因子 (s/h)
        
        # ⚡ LED驱动器规格参数
        self.constant_current = 1.050  # A (1050 mA)
        self.max_power_per_driver = 75.0  # W per driver
        self.red_voltage = 35.0   # V per board
        self.blue_voltage = 45.0  # V per board
        self.boards_per_channel = 1  # 每通道2块板
        
        # 🔴🔵 计算每个通道的实际参数
        self.red_voltage_total = self.red_voltage * self.boards_per_channel   # 70V
        self.blue_voltage_total = self.blue_voltage * self.boards_per_channel # 90V
        self.red_power_per_channel = self.red_voltage_total * self.constant_current   # 73.5W
        self.blue_power_per_channel = self.blue_voltage_total * self.constant_current # 94.5W
        
        # 📊 LED光量子效率 (基于实测数据)
        # 这些值需要根据具体LED芯片规格调整
        self.red_efficiency = 2.8   # μmol/J (典型红光660nm LED)
        self.blue_efficiency = 2.4  # μmol/J (典型蓝光450nm LED)
        
        # ⚙️ 系统效率
        self.driver_efficiency = 0.92  # 92% 驱动器效率
        self.thermal_efficiency = 0.95  # 95% 热管理效率
        self.optical_efficiency = 0.90  # 90% 光学效率
        self.system_efficiency = self.driver_efficiency * self.thermal_efficiency * self.optical_efficiency
        
        self._print_initialization()
    
    def _print_initialization(self):
        """打印初始化信息"""
        print("🌱 详细CLED计算器初始化完成")
        print("=" * 60)
        print(f"📊 环境参数:")
        print(f"   碳排因子 Ca = {self.Ca} kg CO₂/MWh")
        print(f"   照射面积 S = {self.S} m²")
        print(f"   转换因子 = {self.conversion_factor} s/h")
        
        print(f"\n⚡ LED驱动器规格:")
        print(f"   恒定电流: {self.constant_current*1000:.0f} mA")
        print(f"   最大功率: {self.max_power_per_driver} W per driver")
        print(f"   每通道板数: {self.boards_per_channel}")
        
        print(f"\n🔴 红光通道 (660nm):")
        print(f"   单板电压: {self.red_voltage} V")
        print(f"   总电压: {self.red_voltage_total} V ({self.boards_per_channel} 块板)")
        print(f"   通道功率: {self.red_power_per_channel:.1f} W")
        print(f"   光量子效率: {self.red_efficiency} μmol/J")
        
        print(f"\n🔵 蓝光通道 (450nm):")
        print(f"   单板电压: {self.blue_voltage} V")
        print(f"   总电压: {self.blue_voltage_total} V ({self.boards_per_channel} 块板)")
        print(f"   通道功率: {self.blue_power_per_channel:.1f} W")
        print(f"   光量子效率: {self.blue_efficiency} μmol/J")
        
        print(f"\n⚙️ 系统效率:")
        print(f"   驱动器效率: {self.driver_efficiency*100:.0f}%")
        print(f"   热管理效率: {self.thermal_efficiency*100:.0f}%")
        print(f"   光学效率: {self.optical_efficiency*100:.0f}%")
        print(f"   总系统效率: {self.system_efficiency*100:.1f}%")
    
    def calculate_power_consumption(self, ppfd_red: float, ppfd_blue: float) -> Dict[str, float]:
        """
        根据所需PPFD计算实际功率消耗
        
        Parameters
        ----------
        ppfd_red : float
            红光PPFD (μmol·m⁻²·s⁻¹)
        ppfd_blue : float
            蓝光PPFD (μmol·m⁻²·s⁻¹)
            
        Returns
        -------
        Dict[str, float]
            功率消耗详情
        """
        # 计算所需的光量子流量 (μmol/s per m²)
        red_photon_flux = ppfd_red  # μmol·m⁻²·s⁻¹
        blue_photon_flux = ppfd_blue  # μmol·m⁻²·s⁻¹
        
        # 计算所需能量 (J/s per m² = W/m²)
        red_energy_density = red_photon_flux / self.red_efficiency   # W/m²
        blue_energy_density = blue_photon_flux / self.blue_efficiency # W/m²
        
        # 考虑系统效率的实际功率消耗
        red_actual_power = red_energy_density / self.system_efficiency   # W/m²
        blue_actual_power = blue_energy_density / self.system_efficiency # W/m²
        
        # 计算需要的通道数（基于单通道最大功率）
        red_channels_needed = max(1, red_actual_power / self.red_power_per_channel)
        blue_channels_needed = max(1, blue_actual_power / self.blue_power_per_channel)
        
        total_power = red_actual_power + blue_actual_power
        
        return {
            'red_ppfd': ppfd_red,
            'blue_ppfd': ppfd_blue,
            'red_photon_flux': red_photon_flux,
            'blue_photon_flux': blue_photon_flux,
            'red_energy_density': red_energy_density,
            'blue_energy_density': blue_energy_density,
            'red_actual_power': red_actual_power,
            'blue_actual_power': blue_actual_power,
            'red_channels_needed': red_channels_needed,
            'blue_channels_needed': blue_channels_needed,
            'total_power': total_power,
            'power_per_channel_red': self.red_power_per_channel,
            'power_per_channel_blue': self.blue_power_per_channel
        }
    
    def decompose_light(self, ppfd_total: float, rb_ratio: float) -> Tuple[float, float]:
        """
        分解总PPFD为红光和蓝光分量
        
        Parameters
        ----------
        ppfd_total : float
            总PPFD (μmol·m⁻²·s⁻¹)
        rb_ratio : float
            R:B比例 (红光占比)
            
        Returns
        -------
        Tuple[float, float]
            (红光PPFD, 蓝光PPFD)
        """
        red_ppfd = ppfd_total * rb_ratio
        blue_ppfd = ppfd_total * (1 - rb_ratio)
        return red_ppfd, blue_ppfd
    
    def calculate_cled_detailed(self, ppfd: float, rb_ratio: float) -> Dict:
        """
        详细计算CLED，包含所有中间步骤
        
        Parameters
        ----------
        ppfd : float
            总PPFD (μmol·m⁻²·s⁻¹)
        rb_ratio : float
            R:B比例
            
        Returns
        -------
        Dict
            详细计算结果
        """
        result = {
            'input': {'ppfd': ppfd, 'rb_ratio': rb_ratio},
            'light_decomposition': {},
            'power_calculation': {},
            'carbon_calculation': {},
            'cled': 0.0,
            'efficiency_analysis': {}
        }
        
        if ppfd <= 0:
            return result
        
        # 1️⃣ 光谱分解
        red_ppfd, blue_ppfd = self.decompose_light(ppfd, rb_ratio)
        result['light_decomposition'] = {
            'red_ppfd': red_ppfd,
            'blue_ppfd': blue_ppfd,
            'red_percentage': rb_ratio * 100,
            'blue_percentage': (1 - rb_ratio) * 100
        }
        
        # 2️⃣ 功率计算
        power_details = self.calculate_power_consumption(red_ppfd, blue_ppfd)
        result['power_calculation'] = power_details
        
        # 3️⃣ 碳排放计算
        # 转换碳排因子单位: kg CO₂/MWh → g CO₂/kWh
        Ca_g_per_kwh = self.Ca * 0.001
        
        # 计算每小时碳排放 (g CO₂/h per m²)
        red_carbon_emission = (power_details['red_actual_power'] * Ca_g_per_kwh) / 1000  # g CO₂/(h·m²)
        blue_carbon_emission = (power_details['blue_actual_power'] * Ca_g_per_kwh) / 1000  # g CO₂/(h·m²)
        total_carbon_emission = red_carbon_emission + blue_carbon_emission  # g CO₂/(h·m²)
        
        # 转换为 mg·m⁻²·s⁻¹
        cled = total_carbon_emission * 1000 / self.conversion_factor  # mg·m⁻²·s⁻¹
        
        result['carbon_calculation'] = {
            'carbon_factor_g_per_kwh': Ca_g_per_kwh,
            'red_carbon_emission_g_per_h_per_m2': red_carbon_emission,
            'blue_carbon_emission_g_per_h_per_m2': blue_carbon_emission,
            'total_carbon_emission_g_per_h_per_m2': total_carbon_emission,
            'conversion_factor': self.conversion_factor
        }
        
        result['cled'] = cled
        
        # 4️⃣ 效率分析
        total_energy_theoretical = (red_ppfd / self.red_efficiency) + (blue_ppfd / self.blue_efficiency)
        efficiency_loss = power_details['total_power'] - total_energy_theoretical
        
        result['efficiency_analysis'] = {
            'theoretical_energy_demand': total_energy_theoretical,
            'actual_power_consumption': power_details['total_power'],
            'efficiency_loss': efficiency_loss,
            'overall_efficiency': self.system_efficiency,
            'energy_efficiency_ratio': total_energy_theoretical / power_details['total_power'] if power_details['total_power'] > 0 else 0
        }
        
        return result
    
    def calculate_cled(self, ppfd: float, rb_ratio: float) -> float:
        """
        简化版CLED计算（仅返回结果值）
        
        Parameters
        ----------
        ppfd : float
            总PPFD (μmol·m⁻²·s⁻¹)
        rb_ratio : float
            R:B比例
            
        Returns
        -------
        float
            CLED (mg·m⁻²·s⁻¹)
        """
        if ppfd <= 0:
            return 0.0
        
        red_ppfd, blue_ppfd = self.decompose_light(ppfd, rb_ratio)
        power_details = self.calculate_power_consumption(red_ppfd, blue_ppfd)
        
        # 碳排放计算
        Ca_g_per_kwh = self.Ca * 0.001
        total_carbon_emission = (power_details['total_power'] * Ca_g_per_kwh) / 1000
        cled = total_carbon_emission * 1000 / self.conversion_factor
        
        return cled
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理整个数据框，计算所有数据点的CLED
        
        Parameters
        ----------
        df : pd.DataFrame
            包含PPFD, R:B列的数据框
            
        Returns
        -------
        pd.DataFrame
            添加了详细计算列的数据框
        """
        print("🔄 开始批量计算详细CLED...")
        
        # 计算CLED
        df['CLED_Detailed'] = df.apply(
            lambda row: self.calculate_cled(row['PPFD'], row['R:B']), 
            axis=1
        )
        
        # 计算功率分量
        red_powers = []
        blue_powers = []
        total_powers = []
        
        for _, row in df.iterrows():
            red_ppfd, blue_ppfd = self.decompose_light(row['PPFD'], row['R:B'])
            power_details = self.calculate_power_consumption(red_ppfd, blue_ppfd)
            red_powers.append(power_details['red_actual_power'])
            blue_powers.append(power_details['blue_actual_power'])
            total_powers.append(power_details['total_power'])
        
        df['Red_Power_Density'] = red_powers  # W/m²
        df['Blue_Power_Density'] = blue_powers  # W/m²
        df['Total_Power_Density'] = total_powers  # W/m²
        
        print(f"✅ 完成！处理了 {len(df)} 个数据点")
        
        return df
    
    def compare_with_original(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        与原始CLED计算进行比较
        
        Parameters
        ----------
        df : pd.DataFrame
            包含原始CLED列的数据框
            
        Returns
        -------
        pd.DataFrame
            添加了比较分析的数据框
        """
        if 'CLED' not in df.columns:
            print("⚠️  原始CLED列不存在，跳过比较")
            return df
        
        # 计算差异
        df['CLED_Difference'] = df['CLED_Detailed'] - df['CLED']
        df['CLED_Ratio'] = df['CLED_Detailed'] / df['CLED']
        
        print(f"\n📊 CLED比较分析:")
        print(f"原始CLED范围: {df['CLED'].min():.2f} - {df['CLED'].max():.2f}")
        print(f"详细CLED范围: {df['CLED_Detailed'].min():.2f} - {df['CLED_Detailed'].max():.2f}")
        print(f"平均差异: {df['CLED_Difference'].mean():.2f} ± {df['CLED_Difference'].std():.2f}")
        print(f"平均比值: {df['CLED_Ratio'].mean():.3f} ± {df['CLED_Ratio'].std():.3f}")
        
        return df
    
    def analyze_power_efficiency(self, df: pd.DataFrame):
        """分析功率效率"""
        print(f"\n⚡ 功率效率分析:")
        print("=" * 40)
        
        # 总功率统计
        print(f"功率密度范围: {df['Total_Power_Density'].min():.1f} - {df['Total_Power_Density'].max():.1f} W/m²")
        print(f"平均功率密度: {df['Total_Power_Density'].mean():.1f} ± {df['Total_Power_Density'].std():.1f} W/m²")
        
        # 红蓝功率比例分析
        red_power_ratio = df['Red_Power_Density'] / df['Total_Power_Density']
        blue_power_ratio = df['Blue_Power_Density'] / df['Total_Power_Density']
        
        print(f"\n🔴 红光功率占比: {red_power_ratio.mean()*100:.1f}% ± {red_power_ratio.std()*100:.1f}%")
        print(f"🔵 蓝光功率占比: {blue_power_ratio.mean()*100:.1f}% ± {blue_power_ratio.std()*100:.1f}%")
        
        # 不同R:B下的功率效率
        print(f"\n📈 不同R:B比例的功率统计:")
        power_by_rb = df.groupby('R:B')['Total_Power_Density'].agg(['mean', 'std'])
        print(power_by_rb.round(1))

def create_detailed_visualizations(df: pd.DataFrame):
    """创建详细的可视化图表"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Detailed LED Power & Carbon Emissions Analysis', fontsize=16, fontweight='bold')
    
    # 1️⃣ CLED比较
    ax1 = axes[0, 0]
    if 'CLED' in df.columns:
        ax1.scatter(df['CLED'], df['CLED_Detailed'], alpha=0.6)
        min_val = min(df['CLED'].min(), df['CLED_Detailed'].min())
        max_val = max(df['CLED'].max(), df['CLED_Detailed'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax1.set_xlabel('Original CLED (mg·m⁻²·s⁻¹)')
        ax1.set_ylabel('Detailed CLED (mg·m⁻²·s⁻¹)')
        ax1.set_title('CLED Method Comparison')
        ax1.legend()
    else:
        ax1.hist(df['CLED_Detailed'], bins=30, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Detailed CLED (mg·m⁻²·s⁻¹)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Detailed CLED Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2️⃣ 功率密度分布
    ax2 = axes[0, 1]
    ax2.scatter(df['PPFD'], df['Total_Power_Density'], c=df['R:B'], 
               cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('R:B Ratio')
    ax2.set_xlabel('PPFD (μmol·m⁻²·s⁻¹)')
    ax2.set_ylabel('Total Power Density (W/m²)')
    ax2.set_title('Power Density vs PPFD')
    ax2.grid(True, alpha=0.3)
    
    # 3️⃣ 红蓝功率对比
    ax3 = axes[1, 0]
    width = 0.35
    rb_groups = sorted(df['R:B'].unique())
    red_powers = [df[df['R:B']==rb]['Red_Power_Density'].mean() for rb in rb_groups]
    blue_powers = [df[df['R:B']==rb]['Blue_Power_Density'].mean() for rb in rb_groups]
    
    x = np.arange(len(rb_groups))
    ax3.bar(x - width/2, red_powers, width, label='Red Power', color='red', alpha=0.7)
    ax3.bar(x + width/2, blue_powers, width, label='Blue Power', color='blue', alpha=0.7)
    ax3.set_xlabel('R:B Ratio')
    ax3.set_ylabel('Average Power Density (W/m²)')
    ax3.set_title('Red vs Blue Power by R:B Ratio')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{rb:.2f}' for rb in rb_groups])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4️⃣ R:B对CLED的影响
    ax4 = axes[1, 1]
    rb_cled_mean = df.groupby('R:B')['CLED_Detailed'].mean()
    rb_cled_std = df.groupby('R:B')['CLED_Detailed'].std()
    ax4.errorbar(rb_cled_mean.index, rb_cled_mean.values, 
                yerr=rb_cled_std.values, marker='o', capsize=5)
    ax4.set_xlabel('R:B Ratio')
    ax4.set_ylabel('Average CLED (mg·m⁻²·s⁻¹)')
    ax4.set_title('R:B Effect on Detailed CLED')
    ax4.grid(True, alpha=0.3)
    
    # 5️⃣ 功率效率分析
    ax5 = axes[2, 0]
    efficiency_ratio = (df['Red_Power_Density'] + df['Blue_Power_Density']) / df['Total_Power_Density']
    ax5.scatter(df['PPFD'], efficiency_ratio, c=df['R:B'], 
               cmap='plasma', alpha=0.6)
    cbar2 = plt.colorbar(ax5.collections[0], ax=ax5)
    cbar2.set_label('R:B Ratio')
    ax5.set_xlabel('PPFD (μmol·m⁻²·s⁻¹)')
    ax5.set_ylabel('Power Efficiency Ratio')
    ax5.set_title('Power Efficiency vs PPFD')
    ax5.grid(True, alpha=0.3)
    
    # 6️⃣ CLED vs 功率密度关系
    ax6 = axes[2, 1]
    ax6.scatter(df['Total_Power_Density'], df['CLED_Detailed'], 
               c=df['R:B'], cmap='coolwarm', alpha=0.6)
    cbar3 = plt.colorbar(ax6.collections[0], ax=ax6)
    cbar3.set_label('R:B Ratio')
    ax6.set_xlabel('Total Power Density (W/m²)')
    ax6.set_ylabel('Detailed CLED (mg·m⁻²·s⁻¹)')
    ax6.set_title('CLED vs Power Density')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'detailed_cled_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 详细分析图表已保存: {filename}")
    plt.show()

def main():
    """主函数：执行详细CLED计算流程"""
    
    print("🌱 详细LED碳排放计算 (基于实际驱动器规格)")
    print("=" * 70)
    
    # 1️⃣ 加载数据
    print("\n1️⃣ 加载数据...")
    try:
        # 尝试加载带有原始CLED的数据
        df = pd.read_csv('../CLED/averaged_data_with_cled.csv')
        print(f"✅ 成功加载完整数据: {df.shape[0]} 行 × {df.shape[1]} 列")
        has_original_cled = True
    except FileNotFoundError:
        try:
            # 备用：加载原始数据
            df = pd.read_csv('../CLED/averaged_data.csv')
            print(f"✅ 成功加载原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
            has_original_cled = False
        except FileNotFoundError:
            print("❌ 未找到数据文件，请确保文件存在")
            return
    
    print(f"📋 数据列: {list(df.columns)}")
    
    # 检查必要列
    required_cols = ['PPFD', 'R:B']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必要列: {missing_cols}")
        return
    
    # 2️⃣ 初始化详细计算器
    print("\n2️⃣ 初始化详细CLED计算器...")
    calculator = DetailedCLEDCalculator()
    
    # 3️⃣ 计算详细CLED
    print("\n3️⃣ 计算详细CLED...")
    df_with_detailed = calculator.process_dataframe(df.copy())
    
    # 4️⃣ 比较分析（如果有原始CLED）
    if has_original_cled:
        print("\n4️⃣ 比较分析...")
        df_compared = calculator.compare_with_original(df_with_detailed)
    else:
        df_compared = df_with_detailed
    
    # 5️⃣ 功率效率分析
    print("\n5️⃣ 功率效率分析...")
    calculator.analyze_power_efficiency(df_compared)
    
    # 6️⃣ 详细结果分析
    print("\n6️⃣ 详细结果分析...")
    print("=" * 50)
    cled_stats = df_compared['CLED_Detailed'].describe()
    print(f"详细CLED范围: {cled_stats['min']:.2f} - {cled_stats['max']:.2f} mg·m⁻²·s⁻¹")
    print(f"详细CLED平均值: {cled_stats['mean']:.2f} ± {cled_stats['std']:.2f} mg·m⁻²·s⁻¹")
    
    # 7️⃣ 保存结果
    print("\n7️⃣ 保存结果...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'detailed_cled_results_{timestamp}.csv'
    df_compared.to_csv(output_file, index=False)
    print(f"✅ 详细结果已保存: {output_file}")
    
    # 8️⃣ 生成可视化
    print("\n8️⃣ 生成详细可视化...")
    create_detailed_visualizations(df_compared)
    
    # 9️⃣ 显示详细计算示例
    print("\n9️⃣ 详细计算示例:")
    print("-" * 80)
    example_ppfd = 500.0
    example_rb = 0.83
    detailed_result = calculator.calculate_cled_detailed(example_ppfd, example_rb)
    
    print(f"📍 输入参数: PPFD={example_ppfd}, R:B={example_rb}")
    print(f"🔴 红光分量: {detailed_result['light_decomposition']['red_ppfd']:.1f} μmol·m⁻²·s⁻¹")
    print(f"🔵 蓝光分量: {detailed_result['light_decomposition']['blue_ppfd']:.1f} μmol·m⁻²·s⁻¹")
    print(f"🔴 红光功率密度: {detailed_result['power_calculation']['red_actual_power']:.2f} W/m²")
    print(f"🔵 蓝光功率密度: {detailed_result['power_calculation']['blue_actual_power']:.2f} W/m²")
    print(f"⚡ 总功率密度: {detailed_result['power_calculation']['total_power']:.2f} W/m²")
    print(f"🌱 详细CLED: {detailed_result['cled']:.2f} mg·m⁻²·s⁻¹")
    print(f"⚙️ 系统效率: {detailed_result['efficiency_analysis']['overall_efficiency']*100:.1f}%")
    
    # 🔟 驱动器需求分析
    print(f"\n🔟 LED驱动器需求分析:")
    print("-" * 40)
    max_power_idx = df_compared['Total_Power_Density'].idxmax()
    max_power_row = df_compared.loc[max_power_idx]
    power_details = calculator.calculate_power_consumption(
        max_power_row['PPFD'] * max_power_row['R:B'],
        max_power_row['PPFD'] * (1 - max_power_row['R:B'])
    )
    
    print(f"最大功率条件: PPFD={max_power_row['PPFD']}, R:B={max_power_row['R:B']}")
    print(f"需要红光通道: {power_details['red_channels_needed']:.1f} 个")
    print(f"需要蓝光通道: {power_details['blue_channels_needed']:.1f} 个")
    print(f"总功率密度: {power_details['total_power']:.1f} W/m²")
    
    print(f"\n🎉 详细CLED计算完成！")
    print(f"📁 输出文件: {output_file}")
    print(f"📊 可视化图表: detailed_cled_analysis_*.png")
    print(f"⚡ 基于实际LED驱动器规格 (1050mA, 35V红光, 45V蓝光)")

if __name__ == "__main__":
    main() 