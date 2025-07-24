#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成优化流程系统测试脚本
"""

import os
import sys
import logging
import tempfile
from datetime import datetime

# 设置路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_workflow.utils.logging_utils import setup_logging
from integrated_workflow.utils.config_utils import validate_config


def test_config_validation():
    """测试配置验证功能"""
    print("测试配置验证...")
    
    try:
        config_path = 'integrated_workflow/config/workflow_config.yaml'
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            validate_config(config)
            print("✓ 配置验证通过")
        else:
            print("✗ 配置文件不存在")
    except Exception as e:
        print(f"✗ 配置验证失败: {str(e)}")


def test_directory_structure():
    """测试目录结构"""
    print("测试目录结构...")
    
    required_files = [
        'integrated_workflow/config/workflow_config.yaml',
        'integrated_workflow/utils/__init__.py',
        'integrated_workflow/utils/logging_utils.py',
        'integrated_workflow/utils/config_utils.py',
        'integrated_workflow/workflow_manager/__init__.py',
        'integrated_workflow/workflow_manager/integrated_workflow_manager.py',
        'integrated_workflow/run_workflow.py'
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件缺失: {file_path}")
            all_good = False
    
    if all_good:
        print("✓ 目录结构完整")


def main():
    """主测试函数"""
    print("="*60)
    print("集成优化流程系统 - 组件测试")
    print("="*60)
    
    test_directory_structure()
    print()
    test_config_validation()
    print()
    
    print("="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()