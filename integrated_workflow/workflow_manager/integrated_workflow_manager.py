#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成工作流管理器
负责协调ML训练、多目标优化、最优点确定和MPC仿真的执行
"""

import os
import sys
import logging
import subprocess
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class IntegratedWorkflowManager:
    """集成工作流管理器类"""
    
    def __init__(self, config, output_dir):
        """
        初始化工作流管理器
        
        Args:
            config (dict): 工作流配置
            output_dir (str): 输出目录路径
        """
        self.config = config
        self.output_dir = output_dir
        self.results = {}
        self.current_stage = None
        self.start_time = datetime.now()
        
        # 创建检查点目录
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"工作流管理器初始化完成，输出目录: {output_dir}")
    
    def run_complete_workflow(self):
        """
        执行完整的工作流程
        """
        logger.info("开始执行完整工作流...")
        
        try:
            # 1. ML模型训练
            ml_result = self.run_ml_training()
            self.results['ml_training'] = ml_result
            self._save_checkpoint('ml_training')
            
            # 2. 多目标优化
            optimization_result = self.run_optimization(ml_result)
            self.results['optimization'] = optimization_result
            self._save_checkpoint('optimization')
            
            # 3. 最优点确定
            optimal_point_result = self.run_optimal_point_selection(optimization_result)
            self.results['optimal_point'] = optimal_point_result
            self._save_checkpoint('optimal_point')
            
            # 4. MPC仿真
            mpc_result = self.run_mpc_simulation(optimal_point_result)
            self.results['mpc_simulation'] = mpc_result
            self._save_checkpoint('mpc_simulation')
            
            logger.info("完整工作流执行完成")
            return self.results
            
        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}", exc_info=True)
            raise
    
    def run_single_stage(self, stage_name):
        """
        执行单个工作流阶段
        
        Args:
            stage_name (str): 阶段名称 ('ml', 'optimization', 'optimal_point', 'simulation')
        
        Returns:
            dict: 阶段执行结果
        """
        logger.info(f"执行单个阶段: {stage_name}")
        
        if stage_name == 'ml':
            result = self.run_ml_training()
            self.results['ml_training'] = result
        elif stage_name == 'optimization':
            # 需要ML模型结果
            if 'ml_training' not in self.results:
                logger.warning("未找到ML训练结果，尝试加载最近的检查点")
                self._load_latest_checkpoint('ml_training')
            
            result = self.run_optimization(self.results.get('ml_training', {}))
            self.results['optimization'] = result
        elif stage_name == 'optimal_point':
            # 需要优化结果
            if 'optimization' not in self.results:
                logger.warning("未找到优化结果，尝试加载最近的检查点")
                self._load_latest_checkpoint('optimization')
            
            result = self.run_optimal_point_selection(self.results.get('optimization', {}))
            self.results['optimal_point'] = result
        elif stage_name == 'simulation':
            # 需要最优点结果
            if 'optimal_point' not in self.results:
                logger.warning("未找到最优点结果，尝试加载最近的检查点")
                self._load_latest_checkpoint('optimal_point')
            
            result = self.run_mpc_simulation(self.results.get('optimal_point', {}))
            self.results['mpc_simulation'] = result
        else:
            raise ValueError(f"未知的阶段名称: {stage_name}")
        
        self._save_checkpoint(stage_name)
        return result
    
    def run_from_checkpoint(self, checkpoint_path):
        """
        从检查点恢复执行
        
        Args:
            checkpoint_path (str): 检查点文件路径
        """
        logger.info(f"从检查点恢复执行: {checkpoint_path}")
        
        # 加载检查点
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.results = checkpoint_data['results']
        last_completed_stage = checkpoint_data['last_completed_stage']
        
        logger.info(f"已完成的最后阶段: {last_completed_stage}")
        
        # 确定下一个要执行的阶段
        stages = ['ml_training', 'optimization', 'optimal_point', 'mpc_simulation']
        try:
            next_stage_idx = stages.index(last_completed_stage) + 1
        except ValueError:
            next_stage_idx = 0
        
        # 执行剩余阶段
        if next_stage_idx >= len(stages):
            logger.info("所有阶段已完成，无需继续执行")
            return self.results
        
        for stage_idx in range(next_stage_idx, len(stages)):
            stage = stages[stage_idx]
            logger.info(f"执行阶段: {stage}")
            
            if stage == 'ml_training':
                self.results['ml_training'] = self.run_ml_training()
            elif stage == 'optimization':
                self.results['optimization'] = self.run_optimization(self.results['ml_training'])
            elif stage == 'optimal_point':
                self.results['optimal_point'] = self.run_optimal_point_selection(self.results['optimization'])
            elif stage == 'mpc_simulation':
                self.results['mpc_simulation'] = self.run_mpc_simulation(self.results['optimal_point'])
            
            self._save_checkpoint(stage)
        
        logger.info("从检查点恢复执行完成")
        return self.results
    
    def run_ml_training(self):
        """
        执行ML模型训练阶段
        
        Returns:
            dict: 训练结果，包含模型路径和性能指标
        """
        self.current_stage = 'ml_training'
        logger.info("开始ML模型训练阶段...")
        
        ml_config = self.config.get('ml_training', {})
        framework_path = ml_config.get('framework_path', 'ML_Framework')
        script_path = os.path.join(framework_path, ml_config.get('script_path', 'run_experiment.py'))
        config_file = os.path.join(framework_path, ml_config.get('config_file', 'config/config.yaml'))
        
        # 确保脚本存在
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"ML训练脚本不存在: {script_path}")
        
        logger.info(f"执行ML训练脚本: {script_path}")
        logger.info(f"使用配置文件: {config_file}")
        
        # 执行ML训练脚本
        cmd = [sys.executable, script_path, '--config', config_file]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 实时获取输出
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"ML训练失败，返回码: {process.returncode}")
                logger.error(f"错误输出: {stderr}")
                raise RuntimeError(f"ML训练执行失败: {stderr}")
            
            logger.info("ML训练完成")
            
            # 查找最新的结果目录
            results_dir = os.path.join(framework_path, 'results')
            if not os.path.exists(results_dir):
                raise FileNotFoundError(f"ML训练结果目录不存在: {results_dir}")
            
            # 获取最新的结果目录
            result_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
            if not result_folders:
                raise FileNotFoundError(f"未找到ML训练结果")
            
            # 按时间排序，获取最新的
            result_folders.sort(reverse=True)
            latest_result = os.path.join(results_dir, result_folders[0])
            
            logger.info(f"找到最新的ML训练结果: {latest_result}")
            
            # 读取结果信息
            results_json = os.path.join(latest_result, 'results.json')
            if os.path.exists(results_json):
                with open(results_json, 'r') as f:
                    results_data = json.load(f)
            else:
                results_data = {}
            
            # 查找模型文件
            models_dir = os.path.join(latest_result, 'models')
            model_files = []
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            
            # 复制结果到工作流输出目录
            ml_output_dir = os.path.join(self.output_dir, 'ml_training')
            os.makedirs(ml_output_dir, exist_ok=True)
            
            for item in os.listdir(latest_result):
                src = os.path.join(latest_result, item)
                dst = os.path.join(ml_output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            # 构建结果
            result = {
                'status': 'success',
                'result_dir': latest_result,
                'output_dir': ml_output_dir,
                'model_files': model_files,
                'metrics': results_data.get('metrics', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # 查找最佳模型
            best_model_path = None
            for model_file in model_files:
                if 'best' in model_file.lower():
                    best_model_path = os.path.join(models_dir, model_file)
                    break
            
            if best_model_path:
                result['best_model_path'] = best_model_path
                logger.info(f"找到最佳模型: {best_model_path}")
            else:
                logger.warning("未找到标记为'best'的模型文件")
                if model_files:
                    # 使用第一个模型文件
                    best_model_path = os.path.join(models_dir, model_files[0])
                    result['best_model_path'] = best_model_path
                    logger.info(f"使用第一个模型文件作为最佳模型: {best_model_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"ML训练阶段失败: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_optimization(self, ml_result):
        """
        执行多目标优化阶段
        
        Args:
            ml_result (dict): ML训练阶段的结果
        
        Returns:
            dict: 优化结果，包含Pareto解集
        """
        self.current_stage = 'optimization'
        logger.info("开始多目标优化阶段...")
        
        # 检查ML结果
        if ml_result.get('status') != 'success':
            logger.error("ML训练阶段未成功完成，无法继续优化")
            raise RuntimeError("ML训练阶段未成功完成，无法继续优化")
        
        best_model_path = ml_result.get('best_model_path')
        if not best_model_path or not os.path.exists(best_model_path):
            logger.error(f"最佳模型文件不存在: {best_model_path}")
            raise FileNotFoundError(f"最佳模型文件不存在: {best_model_path}")
        
        opt_config = self.config.get('optimization', {})
        framework_path = opt_config.get('framework_path', 'pymoo')
        script_path = os.path.join(framework_path, opt_config.get('script_path', 'find_optimal_conditions_multi_model.py'))
        
        # 确保脚本存在
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"优化脚本不存在: {script_path}")
        
        logger.info(f"执行多目标优化脚本: {script_path}")
        logger.info(f"使用模型: {best_model_path}")
        
        # 执行优化脚本
        cmd = [sys.executable, script_path, '--model', best_model_path]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 实时获取输出
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"优化失败，返回码: {process.returncode}")
                logger.error(f"错误输出: {stderr}")
                raise RuntimeError(f"优化执行失败: {stderr}")
            
            logger.info("多目标优化完成")
            
            # 查找最新的结果目录
            results_dir = os.path.join(framework_path, 'results')
            if not os.path.exists(results_dir):
                raise FileNotFoundError(f"优化结果目录不存在: {results_dir}")
            
            # 获取最新的结果目录
            result_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
            if not result_folders:
                raise FileNotFoundError(f"未找到优化结果")
            
            # 按时间排序，获取最新的
            result_folders.sort(reverse=True)
            latest_result = os.path.join(results_dir, result_folders[0])
            
            logger.info(f"找到最新的优化结果: {latest_result}")
            
            # 查找Pareto前沿文件
            pareto_file = os.path.join(latest_result, 'pareto_front.csv')
            if not os.path.exists(pareto_file):
                logger.warning(f"未找到Pareto前沿文件: {pareto_file}")
                # 尝试查找其他可能的文件名
                csv_files = [f for f in os.listdir(latest_result) if f.endswith('.csv')]
                if csv_files:
                    pareto_file = os.path.join(latest_result, csv_files[0])
                    logger.info(f"使用替代文件作为Pareto前沿: {pareto_file}")
                else:
                    raise FileNotFoundError("未找到任何CSV结果文件")
            
            # 复制结果到工作流输出目录
            opt_output_dir = os.path.join(self.output_dir, 'optimization')
            os.makedirs(opt_output_dir, exist_ok=True)
            
            for item in os.listdir(latest_result):
                src = os.path.join(latest_result, item)
                dst = os.path.join(opt_output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            # 构建结果
            result = {
                'status': 'success',
                'result_dir': latest_result,
                'output_dir': opt_output_dir,
                'pareto_file': pareto_file,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"优化阶段失败: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_optimal_point_selection(self, optimization_result):
        """
        执行最优点确定阶段
        
        Args:
            optimization_result (dict): 优化阶段的结果
        
        Returns:
            dict: 最优点结果，包含最优PPFD和R:B比例
        """
        self.current_stage = 'optimal_point'
        logger.info("开始最优点确定阶段...")
        
        # 检查优化结果
        if optimization_result.get('status') != 'success':
            logger.error("优化阶段未成功完成，无法继续最优点确定")
            raise RuntimeError("优化阶段未成功完成，无法继续最优点确定")
        
        pareto_file = optimization_result.get('pareto_file')
        if not pareto_file or not os.path.exists(pareto_file):
            logger.error(f"Pareto前沿文件不存在: {pareto_file}")
            raise FileNotFoundError(f"Pareto前沿文件不存在: {pareto_file}")
        
        opt_point_config = self.config.get('optimal_point', {})
        framework_path = opt_point_config.get('framework_path', 'Optimal')
        script_path = os.path.join(framework_path, opt_point_config.get('script_path', 'fit.py'))
        
        # 确保脚本存在
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"最优点确定脚本不存在: {script_path}")
        
        logger.info(f"执行最优点确定脚本: {script_path}")
        logger.info(f"使用Pareto前沿文件: {pareto_file}")
        
        # 执行最优点确定脚本
        cmd = [sys.executable, script_path, '--pareto', pareto_file]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 实时获取输出
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"最优点确定失败，返回码: {process.returncode}")
                logger.error(f"错误输出: {stderr}")
                raise RuntimeError(f"最优点确定执行失败: {stderr}")
            
            logger.info("最优点确定完成")
            
            # 查找最新的结果目录
            results_dir = os.path.join(framework_path, 'results')
            if not os.path.exists(results_dir):
                raise FileNotFoundError(f"最优点确定结果目录不存在: {results_dir}")
            
            # 获取最新的结果目录
            result_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]
            if not result_folders:
                raise FileNotFoundError(f"未找到最优点确定结果")
            
            # 按时间排序，获取最新的
            result_folders.sort(reverse=True)
            latest_result = os.path.join(results_dir, result_folders[0])
            
            logger.info(f"找到最新的最优点确定结果: {latest_result}")
            
            # 查找最优点结果文件
            optimal_params_file = os.path.join(latest_result, 'optimal_params.json')
            if not os.path.exists(optimal_params_file):
                logger.warning(f"未找到最优点参数文件: {optimal_params_file}")
                # 尝试查找其他可能的文件名
                json_files = [f for f in os.listdir(latest_result) if f.endswith('.json')]
                if json_files:
                    optimal_params_file = os.path.join(latest_result, json_files[0])
                    logger.info(f"使用替代文件作为最优点参数: {optimal_params_file}")
                else:
                    raise FileNotFoundError("未找到任何JSON结果文件")
            
            # 读取最优点参数
            with open(optimal_params_file, 'r') as f:
                optimal_params = json.load(f)
            
            # 复制结果到工作流输出目录
            opt_point_output_dir = os.path.join(self.output_dir, 'optimal_point')
            os.makedirs(opt_point_output_dir, exist_ok=True)
            
            for item in os.listdir(latest_result):
                src = os.path.join(latest_result, item)
                dst = os.path.join(opt_point_output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            # 构建结果
            result = {
                'status': 'success',
                'result_dir': latest_result,
                'output_dir': opt_point_output_dir,
                'optimal_params_file': optimal_params_file,
                'optimal_params': optimal_params,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"最优点确定阶段失败: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_mpc_simulation(self, optimal_point_result):
        """
        执行MPC仿真阶段
        
        Args:
            optimal_point_result (dict): 最优点确定阶段的结果
        
        Returns:
            dict: 仿真结果，包含性能指标
        """
        self.current_stage = 'mpc_simulation'
        logger.info("开始MPC仿真阶段...")
        
        # 检查最优点结果
        if optimal_point_result.get('status') != 'success':
            logger.error("最优点确定阶段未成功完成，无法继续MPC仿真")
            raise RuntimeError("最优点确定阶段未成功完成，无法继续MPC仿真")
        
        optimal_params = optimal_point_result.get('optimal_params', {})
        if not optimal_params:
            logger.error("未找到最优参数")
            raise ValueError("未找到最优参数")
        
        # 提取PPFD和R:B比例
        ppfd = optimal_params.get('ppfd')
        r_b_ratio = optimal_params.get('r_b_ratio')
        
        if ppfd is None or r_b_ratio is None:
            logger.error(f"最优参数不完整: PPFD={ppfd}, R:B={r_b_ratio}")
            raise ValueError(f"最优参数不完整: PPFD={ppfd}, R:B={r_b_ratio}")
        
        mpc_config = self.config.get('mpc_simulation', {})
        framework_path = mpc_config.get('framework_path', 'mpc-farming-master')
        script_path = os.path.join(framework_path, mpc_config.get('script_path', 'mpc.py'))
        
        # 确保脚本存在
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"MPC仿真脚本不存在: {script_path}")
        
        logger.info(f"执行MPC仿真脚本: {script_path}")
        logger.info(f"使用最优参数: PPFD={ppfd}, R:B={r_b_ratio}")
        
        # 执行MPC仿真脚本
        cmd = [sys.executable, script_path, '--ppfd', str(ppfd), '--rb', str(r_b_ratio)]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # 实时获取输出
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"MPC仿真失败，返回码: {process.returncode}")
                logger.error(f"错误输出: {stderr}")
                raise RuntimeError(f"MPC仿真执行失败: {stderr}")
            
            logger.info("MPC仿真完成")
            
            # 创建仿真结果目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sim_result_dir = os.path.join(framework_path, f'simulation_{timestamp}')
            os.makedirs(sim_result_dir, exist_ok=True)
            
            # 保存仿真日志
            log_file = os.path.join(framework_path, 'mpc_log.txt')
            if os.path.exists(log_file):
                shutil.copy2(log_file, os.path.join(sim_result_dir, 'mpc_log.txt'))
            
            # 查找仿真生成的图表
            for file in os.listdir(framework_path):
                if file.endswith('.png') and ('mpc' in file.lower() or 'simulation' in file.lower()):
                    shutil.copy2(os.path.join(framework_path, file), os.path.join(sim_result_dir, file))
            
            # 复制结果到工作流输出目录
            mpc_output_dir = os.path.join(self.output_dir, 'mpc_simulation')
            os.makedirs(mpc_output_dir, exist_ok=True)
            
            if os.path.exists(sim_result_dir):
                for item in os.listdir(sim_result_dir):
                    src = os.path.join(sim_result_dir, item)
                    dst = os.path.join(mpc_output_dir, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            
            # 解析仿真日志，提取性能指标
            performance_metrics = self._parse_mpc_log(log_file if os.path.exists(log_file) else None)
            
            # 构建结果
            result = {
                'status': 'success',
                'result_dir': sim_result_dir,
                'output_dir': mpc_output_dir,
                'input_params': {
                    'ppfd': ppfd,
                    'r_b_ratio': r_b_ratio
                },
                'performance_metrics': performance_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"MPC仿真阶段失败: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_report(self, output_path):
        """
        生成工作流执行报告
        
        Args:
            output_path (str): 报告输出路径
        """
        logger.info(f"生成工作流执行报告: {output_path}")
        
        # 计算总执行时间
        end_time = datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()
        
        # 构建报告
        report = {
            'workflow_name': '集成优化流程系统',
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'execution_time_seconds': execution_time,
            'output_directory': self.output_dir,
            'stages': {}
        }
        
        # 添加各阶段结果
        for stage, result in self.results.items():
            # 移除大型数据，只保留关键信息
            stage_summary = {
                'status': result.get('status'),
                'timestamp': result.get('timestamp'),
                'output_dir': result.get('output_dir')
            }
            
            if stage == 'ml_training' and result.get('status') == 'success':
                stage_summary['model_files'] = result.get('model_files', [])
                stage_summary['metrics'] = result.get('metrics', {})
                stage_summary['best_model_path'] = result.get('best_model_path')
            
            elif stage == 'optimization' and result.get('status') == 'success':
                stage_summary['pareto_file'] = result.get('pareto_file')
            
            elif stage == 'optimal_point' and result.get('status') == 'success':
                stage_summary['optimal_params'] = result.get('optimal_params', {})
            
            elif stage == 'mpc_simulation' and result.get('status') == 'success':
                stage_summary['input_params'] = result.get('input_params', {})
                stage_summary['performance_metrics'] = result.get('performance_metrics', {})
            
            report['stages'][stage] = stage_summary
        
        # 写入报告
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"工作流执行报告已保存: {output_path}")
        return report
    
    def _save_checkpoint(self, stage_name):
        """
        保存检查点
        
        Args:
            stage_name (str): 阶段名称
        """
        if not self.config.get('general', {}).get('enable_checkpoints', True):
            return
        
        logger.info(f"保存检查点: {stage_name}")
        
        checkpoint_data = {
            'results': self.results,
            'last_completed_stage': stage_name,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_{stage_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        )
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _load_latest_checkpoint(self, stage_name):
        """
        加载最新的检查点
        
        Args:
            stage_name (str): 阶段名称
        
        Returns:
            bool: 是否成功加载
        """
        logger.info(f"尝试加载最新的检查点: {stage_name}")
        
        if not os.path.exists(self.checkpoint_dir):
            logger.warning(f"检查点目录不存在: {self.checkpoint_dir}")
            return False
        
        # 查找匹配的检查点文件
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(f'checkpoint_{stage_name}_') and f.endswith('.pkl')
        ]
        
        if not checkpoint_files:
            logger.warning(f"未找到阶段 {stage_name} 的检查点")
            return False
        
        # 按时间排序，获取最新的
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoint_files[0])
        
        logger.info(f"找到最新的检查点: {latest_checkpoint}")
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # 更新结果
            if stage_name in checkpoint_data.get('results', {}):
                self.results[stage_name] = checkpoint_data['results'][stage_name]
                logger.info(f"成功加载阶段 {stage_name} 的检查点数据")
                return True
            else:
                logger.warning(f"检查点中未找到阶段 {stage_name} 的数据")
                return False
                
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}", exc_info=True)
            return False
    
    def _parse_mpc_log(self, log_file):
        """
        解析MPC日志文件，提取性能指标
        
        Args:
            log_file (str): 日志文件路径
        
        Returns:
            dict: 性能指标
        """
        metrics = {
            'cumulative_photosynthesis': None,
            'energy_consumption': None,
            'control_stability': None
        }
        
        if not log_file or not os.path.exists(log_file):
            logger.warning("MPC日志文件不存在，无法提取性能指标")
            return metrics
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # 解析累积光合作用
            import re
            pn_match = re.search(r'Cumulative Photosynthesis:\s*([\d.]+)', log_content)
            if pn_match:
                metrics['cumulative_photosynthesis'] = float(pn_match.group(1))
            
            # 解析能耗
            energy_match = re.search(r'Energy Consumption:\s*([\d.]+)', log_content)
            if energy_match:
                metrics['energy_consumption'] = float(energy_match.group(1))
            
            # 解析控制稳定性（如果有）
            stability_match = re.search(r'Control Stability:\s*([\d.]+)', log_content)
            if stability_match:
                metrics['control_stability'] = float(stability_match.group(1))
            
        except Exception as e:
            logger.error(f"解析MPC日志失败: {str(e)}", exc_info=True)
        
        return metrics