#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成优化流程系统演示脚本
展示如何使用系统的各种功能
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_section(title):
    """打印章节标题"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 执行成功")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print("❌ 执行失败")
            if result.stderr:
                print("错误:")
                print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 执行异常: {str(e)}")
        return False

def main():
    """主演示函数"""
    print("🌱 集成优化流程系统演示")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 测试系统组件
    print_section("1. 系统组件测试")
    success = run_command(
        [sys.executable, "integrated_workflow/test_workflow.py"],
        "运行系统组件测试"
    )
    
    if not success:
        print("❌ 系统组件测试失败，请检查安装")
        return
    
    # 2. 配置验证
    print_section("2. 配置验证")
    success = run_command(
        [sys.executable, "integrated_workflow/run_workflow.py", "--dry-run"],
        "验证工作流配置"
    )
    
    if not success:
        print("❌ 配置验证失败，请检查配置文件")
        return
    
    # 3. 显示帮助信息
    print_section("3. 命令行帮助")
    run_command(
        [sys.executable, "integrated_workflow/run_workflow.py", "--help"],
        "显示命令行帮助信息"
    )
    
    # 4. 检查外部依赖
    print_section("4. 外部依赖检查")
    dependencies = [
        ("ML_Framework", "ML_Framework/run_experiment.py"),
        ("pymoo", "pymoo/find_optimal_conditions_multi_model.py"),
        ("Optimal", "Optimal/fit.py"),
        ("mpc-farming-master", "mpc-farming-master/mpc.py")
    ]
    
    print("检查外部依赖:")
    all_deps_ok = True
    for dep_name, dep_path in dependencies:
        if os.path.exists(dep_path):
            print(f"✅ {dep_name}: {dep_path}")
        else:
            print(f"❌ {dep_name}: {dep_path} (不存在)")
            all_deps_ok = False
    
    # 5. 使用示例
    print_section("5. 使用示例")
    
    print("📋 可用的命令示例:")
    examples = [
        ("运行完整工作流", "python integrated_workflow/run_workflow.py"),
        ("只运行ML训练", "python integrated_workflow/run_workflow.py --stage ml"),
        ("只运行优化", "python integrated_workflow/run_workflow.py --stage optimization"),
        ("只运行最优点确定", "python integrated_workflow/run_workflow.py --stage optimal_point"),
        ("只运行MPC仿真", "python integrated_workflow/run_workflow.py --stage simulation"),
        ("从检查点恢复", "python integrated_workflow/run_workflow.py --resume --checkpoint path/to/checkpoint.pkl"),
        ("调试模式", "python integrated_workflow/run_workflow.py --log-level DEBUG"),
        ("验证配置", "python integrated_workflow/run_workflow.py --dry-run")
    ]
    
    for i, (desc, cmd) in enumerate(examples, 1):
        print(f"{i}. {desc}")
        print(f"   {cmd}")
        print()
    
    # 6. 系统状态总结
    print_section("6. 系统状态总结")
    
    print("📊 系统状态:")
    print(f"✅ 集成工作流系统已安装")
    print(f"✅ 配置文件验证通过")
    print(f"✅ 系统组件测试通过")
    
    if all_deps_ok:
        print(f"✅ 所有外部依赖都存在")
        print(f"🚀 系统已准备就绪，可以运行完整工作流！")
    else:
        print(f"⚠️  部分外部依赖缺失")
        print(f"📝 请确保以下组件存在后再运行完整工作流:")
        for dep_name, dep_path in dependencies:
            if not os.path.exists(dep_path):
                print(f"   - {dep_path}")
    
    print("\n📚 更多信息请查看:")
    print("   - README: integrated_workflow/README.md")
    print("   - 配置文件: integrated_workflow/config/workflow_config.yaml")
    print("   - 日志目录: integrated_workflow/logs/")
    print("   - 结果目录: integrated_workflow/results/")
    
    print(f"\n🎉 演示完成！")

if __name__ == "__main__":
    main()