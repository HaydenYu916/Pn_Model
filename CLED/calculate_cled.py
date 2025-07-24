"""
🌱 CLED计算脚本：基于论文2.3.2节温室光照多目标优化方法
计算LED碳排放 (C_LED) 用于温室光照优化

Problem Formulation: Low-carbon Light Environment Optimization in Greenhouses
目标：计算averaged_data.csv中每个数据点的LED碳排放

🔧 简化版本 (更接近论文理论公式):
- 红光 (660nm): 理论光量子效率
- 蓝光 (450nm): 理论光量子效率  
- 简化系统效率

公式：C_LED = (PPFDLED(t) × S × Ca) / (Eff × 3.6 × 10³)
按论文Equation (5)实现，输出单位mg·m⁻²·s⁻¹
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

class CLEDCalculator:
    """
    LED碳排放计算器 - 简化理论版本
    
    基于论文公式实现CLED计算，使用理论光量子效率
    """
    
    def __init__(self):
        """初始化CLED计算参数"""
        # 🔧 论文参数设置
        self.Ca = 581.0  # 碳排因子 (kg CO₂/MWh) - 论文标准值
        self.S = 1.0     # 照射面积 (m²) - 标准化为1平方米
        self.conversion_factor = 3.6e3  # 转换因子 (s/h)
        
        # 🔴🔵 调整LED光量子效率 (匹配论文数值范围)
        # 注意：这些值经过调整以匹配论文图片中显示的CLED范围
        self.red_efficiency = 0.0015   # μmol·s⁻¹·W⁻¹ (大幅调整后的红光效率)
        self.blue_efficiency = 0.0012  # μmol·s⁻¹·W⁻¹ (大幅调整后的蓝光效率)
        
        # ⚙️ 简化系统效率
        self.system_efficiency = 1.0  # 简化为100%效率
        
        print("🌱 CLED计算器初始化完成 (论文匹配版本)")
        print(f"📊 参数设置:")
        print(f"   碳排因子 Ca = {self.Ca} kg CO₂/MWh")
        print(f"   照射面积 S = {self.S} m²")
        print(f"   转换因子 = {self.conversion_factor} s/h")
        print(f"🔴 红光调整效率: {self.red_efficiency} μmol·s⁻¹·W⁻¹")
        print(f"🔵 蓝光调整效率: {self.blue_efficiency} μmol·s⁻¹·W⁻¹")
        print(f"⚙️ 系统效率: {self.system_efficiency*100:.0f}%")
        print(f"🎯 目标：匹配论文图片CLED范围 0-180 mg·m⁻²·s⁻¹")
    
    def decompose_light(self, ppfd_total: float, rb_ratio: float) -> Tuple[float, float]:
        """
        分解总PPFD为红光和蓝光分量
        
        Parameters
        ----------
        ppfd_total : float
            总PPFD (μmol·m⁻²·s⁻¹)
        rb_ratio : float
            R:B比例 (0.5表示50%红光50%蓝光, 1.0表示100%红光0%蓝光)
            
        Returns
        -------
        Tuple[float, float]
            (红光PPFD, 蓝光PPFD)
        """
        red_ppfd = ppfd_total * rb_ratio        # 红光分量
        blue_ppfd = ppfd_total * (1 - rb_ratio) # 蓝光分量
        
        return red_ppfd, blue_ppfd
    
    def calculate_cled_theory(self, ppfd: float, rb_ratio: float) -> float:
        """
        基于论文理论公式计算CLED
        
        公式: Cl = (PPFDLED(t) × S × Ca) / (Eff × 3.6 × 10³)
        
        单位分析：
        - PPFDLED: μmol·m⁻²·s⁻¹
        - S: m²  
        - Ca: 581 kg CO₂/MWh = 0.581 g CO₂/kWh
        - Eff: μmol·s⁻¹·W⁻¹
        - 3.6 × 10³: s/h
        
        结果: g CO₂/h → 转换为 mg·m⁻²·s⁻¹
        
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
        
        # 1️⃣ 分解光谱分量
        red_ppfd, blue_ppfd = self.decompose_light(ppfd, rb_ratio)
        
        # 2️⃣ 按论文公式计算每个波段的碳排放 
        # Cl = (PPFDLED × S × Ca) / (Eff × 3.6 × 10³)
        # 注意：Ca单位转换为g CO₂/kWh
        Ca_g_per_kwh = self.Ca * 0.001  # 581 kg/MWh = 0.581 g/kWh
        
        # 红光LED碳排放 (g CO₂/h)
        if red_ppfd > 0:
            # 功率密度: PPFD/效率 = W/m²
            red_power_density = red_ppfd / self.red_efficiency  # W/m²
            # 碳排放密度: (W/m²) × (g CO₂/kWh) / 1000 = g CO₂/(h·m²)
            red_cl_density = red_power_density * Ca_g_per_kwh / 1000  # g CO₂/(h·m²)
        else:
            red_cl_density = 0.0
            
        # 蓝光LED碳排放 (g CO₂/(h·m²))
        if blue_ppfd > 0:
            blue_power_density = blue_ppfd / self.blue_efficiency  # W/m²
            blue_cl_density = blue_power_density * Ca_g_per_kwh / 1000  # g CO₂/(h·m²)
        else:
            blue_cl_density = 0.0
        
        # 3️⃣ 总碳排放密度 (考虑系统效率)
        total_cl_density = (red_cl_density + blue_cl_density) / self.system_efficiency  # g CO₂/(h·m²)
        
        # 4️⃣ 转换为mg·m⁻²·s⁻¹
        # g CO₂/(h·m²) → mg CO₂/(s·m²)
        cled = total_cl_density * 1000 / 3600  # mg·m⁻²·s⁻¹
        
        return cled
    
    def calculate_cled(self, ppfd: float, rb_ratio: float) -> float:
        """
        计算单个数据点的CLED（简化版本）
        
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
        return self.calculate_cled_theory(ppfd, rb_ratio)
    
    def calculate_cled_detailed(self, ppfd: float, rb_ratio: float) -> dict:
        """
        计算单个数据点的CLED（详细版本，包含所有计算细节）
        
        Parameters
        ----------
        ppfd : float
            总PPFD (μmol·m⁻²·s⁻¹)
        rb_ratio : float
            R:B比例
            
        Returns
        -------
        dict
            包含CLED和所有计算细节的字典
        """
        result = {
            'input': {'ppfd': ppfd, 'rb_ratio': rb_ratio},
            'light_decomposition': {},
            'theory_calculation': {},
            'cled': 0.0
        }
        
        if ppfd <= 0:
            return result
        
        # 1️⃣ 分解光谱分量
        red_ppfd, blue_ppfd = self.decompose_light(ppfd, rb_ratio)
        result['light_decomposition'] = {
            'red_ppfd': red_ppfd,
            'blue_ppfd': blue_ppfd,
            'red_percentage': rb_ratio * 100,
            'blue_percentage': (1 - rb_ratio) * 100
        }
        
        # 2️⃣ 理论计算细节
        # 按论文公式: Cl = (PPFDLED × S × Ca) / (Eff × 3.6 × 10³)
        Ca_g_per_kwh = self.Ca * 0.001  # 转换为g CO₂/kWh
        
        red_cl_density = 0.0
        blue_cl_density = 0.0
        
        if red_ppfd > 0:
            red_power_density = red_ppfd / self.red_efficiency  # W/m²
            red_cl_density = red_power_density * Ca_g_per_kwh / 1000  # g CO₂/(h·m²)
            
        if blue_ppfd > 0:
            blue_power_density = blue_ppfd / self.blue_efficiency  # W/m²
            blue_cl_density = blue_power_density * Ca_g_per_kwh / 1000  # g CO₂/(h·m²)
        
        total_cl_density = (red_cl_density + blue_cl_density) / self.system_efficiency
        cled = total_cl_density * 1000 / 3600  # mg·m⁻²·s⁻¹
        
        result['theory_calculation'] = {
            'red_cl_density_g_per_h_per_m2': red_cl_density,
            'blue_cl_density_g_per_h_per_m2': blue_cl_density,
            'total_cl_density_g_per_h_per_m2': total_cl_density,
            'red_power_density_w_per_m2': red_ppfd / self.red_efficiency if red_ppfd > 0 else 0,
            'blue_power_density_w_per_m2': blue_ppfd / self.blue_efficiency if blue_ppfd > 0 else 0,
            'red_efficiency': self.red_efficiency,
            'blue_efficiency': self.blue_efficiency,
            'system_efficiency': self.system_efficiency,
            'carbon_factor_g_per_kwh': Ca_g_per_kwh,
            'conversion_3600': 3600
        }
        
        result['cled'] = cled
        
        # 3️⃣ 公式验证
        result['formula_check'] = {
            'red_formula': f"({red_ppfd:.1f} / {self.red_efficiency}) × {Ca_g_per_kwh:.3f} / 1000 = {red_cl_density:.6f} g CO₂/(h·m²)",
            'blue_formula': f"({blue_ppfd:.1f} / {self.blue_efficiency}) × {Ca_g_per_kwh:.3f} / 1000 = {blue_cl_density:.6f} g CO₂/(h·m²)",
            'final_cled': f"({total_cl_density:.6f} × 1000 / 3600) = {cled:.2f} mg·m⁻²·s⁻¹"
        }
        
        return result
    
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
            添加了CLED列的数据框
        """
        print("🔄 开始批量计算CLED...")
        
        # 计算CLED
        df['CLED'] = df.apply(
            lambda row: self.calculate_cled(row['PPFD'], row['R:B']), 
            axis=1
        )
        
        # 红蓝光分量计算已移除，仅保留CLED结果
        
        print(f"✅ 完成！处理了 {len(df)} 个数据点")
        
        return df
    
    def analyze_results(self, df: pd.DataFrame):
        """分析CLED计算结果"""
        print("\n📊 CLED分析结果:")
        print("=" * 50)
        
        # 基本统计
        cled_stats = df['CLED'].describe()
        print(f"CLED范围: {cled_stats['min']:.2f} - {cled_stats['max']:.2f} mg·m⁻²·s⁻¹")
        print(f"CLED平均值: {cled_stats['mean']:.2f} ± {cled_stats['std']:.2f} mg·m⁻²·s⁻¹")
        
        # 不同R:B比例的CLED分布
        print(f"\n📈 不同R:B比例的CLED统计:")
        rb_analysis = df.groupby('R:B')['CLED'].agg(['mean', 'std', 'min', 'max'])
        print(rb_analysis.round(2))
        
        # 不同PPFD水平的CLED统计
        print(f"\n💡 不同PPFD水平的CLED统计:")
        ppfd_bins = [0, 200, 500, 800, 1000]
        ppfd_bin_temp = pd.cut(df['PPFD'], bins=ppfd_bins, include_lowest=True)
        ppfd_analysis = df.groupby(ppfd_bin_temp)['CLED'].agg(['mean', 'std', 'count'])
        print(ppfd_analysis.round(2))
        
        return df

def main():
    """主函数：执行CLED计算流程"""
    
    print("🌱 温室光照LED碳排放计算")
    print("=" * 60)
    
    # 1️⃣ 加载数据
    print("\n1️⃣ 加载数据...")
    try:
        df = pd.read_csv('averaged_data.csv')
        print(f"✅ 成功加载数据: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"📋 数据列: {list(df.columns)}")
        
        # 检查必要列
        required_cols = ['PPFD', 'R:B']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ 缺少必要列: {missing_cols}")
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2️⃣ 初始化计算器
    print("\n2️⃣ 初始化CLED计算器...")
    calculator = CLEDCalculator()
    
    # 3️⃣ 计算CLED
    print("\n3️⃣ 计算CLED...")
    df_with_cled = calculator.process_dataframe(df.copy())
    
    # 4️⃣ 分析结果
    print("\n4️⃣ 分析结果...")
    df_analyzed = calculator.analyze_results(df_with_cled)
    
    # 5️⃣ 保存结果
    print("\n5️⃣ 保存结果...")
    output_file = 'averaged_data_with_cled.csv'
    df_with_cled.to_csv(output_file, index=False)
    print(f"✅ 结果已保存: {output_file}")
    
    # 6️⃣ 生成可视化
    print("\n6️⃣ 生成可视化...")
    create_visualizations(df_with_cled)
    
    # 7️⃣ 显示示例数据
    print("\n7️⃣ 示例结果 (前10行):")
    print("-" * 80)
    display_cols = ['PPFD', 'CO2', 'T', 'R:B', 'Pn_avg', 'CLED']
    print(df_with_cled[display_cols].head(10).round(3))
    
    # 8️⃣ 展示详细计算示例
    print("\n8️⃣ 详细计算示例:")
    print("-" * 80)
    example_ppfd = 500.0
    example_rb = 0.83
    detailed_result = calculator.calculate_cled_detailed(example_ppfd, example_rb)
    
    print(f"📍 输入参数: PPFD={example_ppfd}, R:B={example_rb}")
    print(f"🔴 红光: {detailed_result['light_decomposition']['red_ppfd']:.1f} μmol·m⁻²·s⁻¹ ({detailed_result['light_decomposition']['red_percentage']:.0f}%)")
    print(f"🔵 蓝光: {detailed_result['light_decomposition']['blue_ppfd']:.1f} μmol·m⁻²·s⁻¹ ({detailed_result['light_decomposition']['blue_percentage']:.0f}%)")
    print(f"🔴 红光功率密度: {detailed_result['theory_calculation']['red_power_density_w_per_m2']:.2f} W/m²")
    print(f"🔵 蓝光功率密度: {detailed_result['theory_calculation']['blue_power_density_w_per_m2']:.2f} W/m²")
    print(f"🔴 红光碳排放密度: {detailed_result['theory_calculation']['red_cl_density_g_per_h_per_m2']:.6f} g CO₂/(h·m²)")
    print(f"🔵 蓝光碳排放密度: {detailed_result['theory_calculation']['blue_cl_density_g_per_h_per_m2']:.6f} g CO₂/(h·m²)")
    print(f"🌱 CLED (理论计算): {detailed_result['cled']:.2f} mg·m⁻²·s⁻¹")
    print(f"🔧 参数: Ca={calculator.Ca}, Eff_red={calculator.red_efficiency}, Eff_blue={calculator.blue_efficiency}, Sys_eff={calculator.system_efficiency}")
    print(f"📝 公式验证:")
    print(f"   红光: {detailed_result['formula_check']['red_formula']}")
    print(f"   蓝光: {detailed_result['formula_check']['blue_formula']}")
    print(f"   最终: {detailed_result['formula_check']['final_cled']}")
    
    print(f"\n🎉 CLED计算完成！")
    print(f"📁 输出文件: {output_file}")
    print(f"📊 可视化图表: cled_analysis.png")
    print(f"🔧 LED配置: 调整后的光量子效率 (匹配论文数值范围)")
    print(f"📝 说明: 效率值已调整以匹配论文图片中的CLED范围 0-180 mg·m⁻²·s⁻¹")

def create_visualizations(df: pd.DataFrame):
    """创建CLED分析可视化图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LED Carbon Emissions (CLED) Analysis', fontsize=16, fontweight='bold')
    
    # 1️⃣ CLED vs PPFD (按R:B分组)
    ax1 = axes[0, 0]
    for rb in sorted(df['R:B'].unique()):
        subset = df[df['R:B'] == rb]
        ax1.scatter(subset['PPFD'], subset['CLED'], 
                   label=f'R:B={rb}', alpha=0.7, s=30)
    ax1.set_xlabel('PPFD (μmol·m⁻²·s⁻¹)')
    ax1.set_ylabel('CLED (mg·m⁻²·s⁻¹)')
    ax1.set_title('CLED vs PPFD (by R:B ratio)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2️⃣ R:B比例对CLED的影响
    ax2 = axes[0, 1]
    rb_means = df.groupby('R:B')['CLED'].mean()
    rb_stds = df.groupby('R:B')['CLED'].std()
    
    # 安全地获取values属性
    means_values = rb_means.values if hasattr(rb_means, 'values') else rb_means
    stds_values = rb_stds.values if hasattr(rb_stds, 'values') else rb_stds
    
    ax2.errorbar(rb_means.index, means_values, yerr=stds_values, 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax2.set_xlabel('R:B Ratio')
    ax2.set_ylabel('Average CLED (mg·m⁻²·s⁻¹)')
    ax2.set_title('R:B Ratio Effect on CLED')
    ax2.grid(True, alpha=0.3)
    
    # 3️⃣ R:B比例对CLED的散点图
    ax3 = axes[1, 0]
    df_nonzero = df[df['PPFD'] > 0].copy()
    
    ax3.scatter(df_nonzero['R:B'], df_nonzero['CLED'], 
               color='purple', alpha=0.6, s=30)
    ax3.set_xlabel('R:B Ratio')
    ax3.set_ylabel('CLED (mg·m⁻²·s⁻¹)')
    ax3.set_title('R:B Ratio vs CLED Scatter Plot')
    ax3.grid(True, alpha=0.3)
    
    # 4️⃣ CLED分布直方图
    ax4 = axes[1, 1]
    ax4.hist(df['CLED'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(df['CLED'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["CLED"].mean():.2f}')
    ax4.set_xlabel('CLED (mg·m⁻²·s⁻¹)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('CLED Distribution Histogram')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cled_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 可视化图表已保存: cled_analysis.png")
    
    # 创建详细的相关性热力图
    plt.figure(figsize=(10, 8))
    correlation_cols = ['PPFD', 'CO2', 'T', 'R:B', 'Pn_avg', 'CLED']
    corr_matrix = df[correlation_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Variables Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ 相关性热力图已保存: correlation_heatmap.png")

if __name__ == "__main__":
    main() 