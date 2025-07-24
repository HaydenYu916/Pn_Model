#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成优化流程系统主程序
该脚本协调执行从ML模型训练到MPC仿真的完整工作流程
"""

import os
import sys
import argparse
import logging
import yaml
import time
import json
from datetime import datetime
from pathlib import Path

# 设置路径，确保可以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工作流组件
from integrated_workflow.workflow_manager.integrated_workflow_manager import IntegratedWorkflowManager
from integrated_workflow.utils.logging_utils import setup_logging
from integrated_workflow.utils.config_utils import validate_config, load_config_with_defaults


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='集成优化流程系统 - 整合ML训练、多目标优化、最优点确定和MPC仿真',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行完整工作流
  python integrated_workflow/run_workflow.py
  
  # 运行单个阶段
  python integrated_workflow/run_workflow.py --stage ml
  
  # 从检查点恢复
  python integrated_workflow/run_workflow.py --resume --checkpoint checkpoints/checkpoint_ml_training.pkl
  
  # 使用自定义配置
  python integrated_workflow/run_workflow.py --config my_config.yaml
  
  # 仅验证配置
  python integrated_workflow/run_workflow.py --dry-run
  
  # 生成默认配置
  python integrated_workflow/run_workflow.py --generate-config my_config.yaml
        """
    )
    
    parser.add_argument('--config', type=str, 
                        default='integrated_workflow/config/workflow_config.yaml',
                        help='配置文件路径 (默认: integrated_workflow/config/workflow_config.yaml)')
    
    parser.add_argument('--stage', type=str, 
                        choices=['ml', 'optimization', 'optimal_point', 'simulation', 'all'],
                        default='all', 
                        help='要执行的工作流阶段 (默认: all)')
    
    parser.add_argument('--resume', action='store_true',
                        help='从检查点恢复执行')
    
    parser.add_argument('--checkpoint', type=str,
                        help='检查点文件路径（与--resume一起使用）')
    
    parser.add_argument('--log-level', type=str, 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', 
                        help='日志级别 (默认: INFO)')
    
    parser.add_argument('--output-dir', type=str,
                        help='输出目录路径（覆盖配置文件中的设置）')
    
    parser.add_argument('--dry-run', action='store_true',
                        help='仅验证配置，不执行工作流')
    
    parser.add_argument('--generate-config', type=str, metavar='FILE',
                        help='生成默认配置文件并保存到指定路径')
    
    parser.add_argument('--force', action='store_true',
                        help='强制执行，忽略警告')
    
    parser.add_argument('--no-banner', action='store_true',
                        help='不显示程序横幅')
    
    return parser.parse_args()


def print_banner():
    """打印程序横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    集成优化流程系统                          ║
    ║                Integrated Optimization Pipeline              ║
    ║                                                              ║
    ║  ML训练 → 多目标优化 → 最优点确定 → MPC仿真                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 显示横幅
    if not args.no_banner:
        print_banner()
    
    # 处理生成配置文件的请求
    if args.generate_config:
        from integrated_workflow.utils.config_utils import get_default_config, save_config_copy
        
        print(f"生成默认配置文件: {args.generate_config}")
        default_config = get_default_config()
        
        # 确保目录存在
        config_dir = os.path.dirname(args.generate_config)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        # 保存配置
        save_config_copy(default_config, args.generate_config)
        print(f"默认配置已保存到: {args.generate_config}")
        sys.exit(0)
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print("请确保配置文件存在，或使用以下选项:")
        print("  --config 指定正确的配置文件路径")
        print("  --generate-config 生成默认配置文件")
        sys.exit(1)
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        result_dir = args.output_dir
    else:
        result_dir = os.path.join('integrated_workflow/results', f'workflow_{timestamp}')
    
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(result_dir, 'workflow.log')
    setup_logging(log_file, args.log_level)
    logger = logging.getLogger(__name__)
    
    # 记录系统信息
    from integrated_workflow.utils.logging_utils import log_system_info
    log_system_info(logger)
    
    logger.info("="*60)
    logger.info("集成优化流程系统启动")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"执行阶段: {args.stage}")
    logger.info(f"结果目录: {result_dir}")
    logger.info(f"日志级别: {args.log_level}")
    if args.resume:
        logger.info(f"恢复模式: 是, 检查点: {args.checkpoint}")
    if args.dry_run:
        logger.info("仅验证配置模式: 是")
    if args.force:
        logger.info("强制执行模式: 是")
    logger.info("="*60)
    
    try:
        # 加载配置
        logger.info("加载配置文件...")
        from integrated_workflow.utils.config_utils import load_config_with_defaults, validate_config, save_config_copy
        config = load_config_with_defaults(args.config)
        
        # 如果指定了输出目录，覆盖配置
        if args.output_dir:
            config['general']['output_dir'] = args.output_dir
        
        # 验证配置
        logger.info("验证配置...")
        validate_config(config)
        
        # 保存配置副本到结果目录
        config_copy_path = os.path.join(result_dir, 'config.yaml')
        save_config_copy(config, config_copy_path)
        
        # 如果是仅验证模式，到此结束
        if args.dry_run:
            logger.info("配置验证通过，仅验证模式，退出执行")
            print("\n配置验证通过!")
            print(f"配置文件: {args.config}")
            print("所有必需的配置项都已正确设置")
            sys.exit(0)
        
        # 创建工作流管理器
        logger.info("初始化工作流管理器...")
        from integrated_workflow.workflow_manager.integrated_workflow_manager import IntegratedWorkflowManager
        workflow_manager = IntegratedWorkflowManager(config, result_dir)
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行工作流
        if args.resume:
            if not args.checkpoint:
                logger.error("使用--resume时必须指定--checkpoint参数")
                print("错误: 使用--resume时必须指定--checkpoint参数")
                sys.exit(1)
            
            if not os.path.exists(args.checkpoint):
                logger.error(f"检查点文件不存在: {args.checkpoint}")
                print(f"错误: 检查点文件不存在: {args.checkpoint}")
                sys.exit(1)
            
            logger.info(f"从检查点恢复执行: {args.checkpoint}")
            print(f"从检查点恢复执行: {args.checkpoint}")
            results = workflow_manager.run_from_checkpoint(args.checkpoint)
        else:
            if args.stage == 'all':
                logger.info("执行完整工作流...")
                print("执行完整工作流...")
                results = workflow_manager.run_complete_workflow()
            else:
                logger.info(f"执行单个阶段: {args.stage}")
                print(f"执行单个阶段: {args.stage}")
                results = workflow_manager.run_single_stage(args.stage)
        
        # 计算执行时间
        from integrated_workflow.utils.logging_utils import log_execution_time
        execution_time = log_execution_time(logger, start_time, "工作流")
        
        # 生成最终报告
        logger.info("生成工作流执行报告...")
        report_path = os.path.join(result_dir, 'workflow_report.json')
        report = workflow_manager.generate_report(report_path)
        
        # 打印执行摘要
        print("\n" + "="*60)
        print("工作流执行完成!")
        print(f"总执行时间: {execution_time:.2f} 秒")
        print(f"结果目录: {result_dir}")
        print(f"执行报告: {report_path}")
        
        # 打印各阶段状态
        print("\n各阶段执行状态:")
        for stage, result in results.items():
            status = result.get('status', 'unknown')
            status_symbol = "✓" if status == 'success' else "✗"
            print(f"  {status_symbol} {stage}: {status}")
            
            if status == 'failed':
                error = result.get('error', '未知错误')
                print(f"    错误: {error}")
        
        # 如果有失败的阶段，退出码为1
        failed_stages = [stage for stage, result in results.items() 
                        if result.get('status') != 'success']
        
        if failed_stages:
            print(f"\n警告: 以下阶段执行失败: {', '.join(failed_stages)}")
            logger.warning(f"部分阶段执行失败: {failed_stages}")
            sys.exit(1)
        else:
            print("\n所有阶段执行成功!")
            logger.info("所有阶段执行成功")
        
        logger.info("="*60)
        logger.info("工作流执行完成")
        logger.info(f"总执行时间: {execution_time:.2f} 秒")
        logger.info(f"结果保存在: {result_dir}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("用户中断执行")
        print("\n用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}", exc_info=True)
        print(f"\n错误: 工作流执行失败")
        print(f"详细错误信息: {str(e)}")
        print(f"请查看日志文件获取更多信息: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()