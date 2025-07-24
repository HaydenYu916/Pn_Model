import numpy as np
import requests
import json
import re
import time
from pymoo.core.crossover import Crossover


class GPT_OpenAI(Crossover):
    """
    基于OpenAI官方API文档标准的GPT交叉算子
    https://platform.openai.com/docs/api-reference/introduction
    """

    def __init__(self, n_new, **kwargs):
        super().__init__(10, 2, **kwargs)
        self.n_new = n_new

    def get_prompt(self, x, y, obj_p):
        """构建提示内容"""
        pop_content = " "
        for i in range(len(x)):
            pop_content += "point: <start>" + ",".join(str(idx) for idx in x[i].tolist()) + "<end> \n objective 1: " + str(round(obj_p[i][0], 4)) + " objective 2: " + str(round(obj_p[i][1], 4)) + "\n\n"
        
        prompt_content = ("Now you will help me minimize " + str(len(obj_p[0])) + " objectives with " + str(len(x[0])) + " variables. "
                         "I have some points with their objective values. The points start with <start> and end with <end>.\n\n"
                         + pop_content +
                         "Give me two new points that are different from all points above, and not dominated by any of the above. "
                         "Do not write code. Do not give any explanation. Each output new point must start with <start> and end with <end>")
        return prompt_content

    def call_openai_api(self, prompt, model_LLM, endpoint, key, max_retries=5):
        """
        根据OpenAI官方文档调用API
        https://platform.openai.com/docs/api-reference/chat/create
        """
        # 构建标准的OpenAI API URL
        if endpoint.startswith('http'):
            url = f"{endpoint}/v1/chat/completions"
        else:
            url = f"https://{endpoint}/v1/chat/completions"
        
        # 标准的OpenAI API headers
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        # 标准的OpenAI API payload
        data = {
            "model": model_LLM,
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 调用API (尝试 {attempt + 1}/{max_retries})...")
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        print(f"✅ API调用成功")
                        return content
                    else:
                        print(f"❌ API响应格式错误: {result}")
                        continue
                        
                elif response.status_code == 401:
                    print(f"❌ API密钥无效: {response.text}")
                    break
                    
                elif response.status_code == 429:
                    print(f"⚠️  API调用限制，等待重试...")
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                    
                else:
                    print(f"❌ API调用失败 ({response.status_code}): {response.text}")
                    time.sleep(2)
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"⏰ API调用超时，重试中...")
                time.sleep(2)
                continue
                
            except requests.exceptions.ConnectionError:
                print(f"🌐 网络连接错误，重试中...")
                time.sleep(2)
                continue
                
            except Exception as e:
                print(f"❌ 未知错误: {str(e)}")
                time.sleep(2)
                continue
        
        print(f"❌ API调用失败，已尝试 {max_retries} 次")
        return None

    def call_ollama_api(self, prompt, model_LLM, endpoint, max_retries=3):
        """
        调用Ollama本地API
        """
        if endpoint.startswith('http'):
            url = f"{endpoint}/api/chat"
        else:
            url = f"http://{endpoint}/api/chat"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_LLM,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 调用Ollama (尝试 {attempt + 1}/{max_retries})...")
                
                response = requests.post(url, headers=headers, json=data, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'message' in result and 'content' in result['message']:
                        content = result['message']['content']
                        print(f"✅ Ollama调用成功")
                        return content
                    else:
                        print(f"❌ Ollama响应格式错误: {result}")
                        continue
                else:
                    print(f"❌ Ollama调用失败 ({response.status_code}): {response.text}")
                    time.sleep(2)
                    continue
                    
            except Exception as e:
                print(f"❌ Ollama调用错误: {str(e)}")
                time.sleep(2)
                continue
        
        print(f"❌ Ollama调用失败，已尝试 {max_retries} 次")
        return None

    def _do(self, _, X, Y, debug_mode, model_LLM, endpoint, key, out_filename, parents_obj, **kwargs):
        """执行交叉操作"""
        
        # 数据预处理
        y_p = np.zeros(len(Y))
        x_p = np.zeros((len(X), len(X[0][0])))
        
        for i in range(len(Y)):
            y_p[i] = round(Y[i][0][0], 4)
            x_p[i] = X[i][0]
            x_p[i] = np.round((x_p[i] - _.xl) / (_.xu - _.xl), 4)

        # 排序
        sort_idx = sorted(range(len(Y)), key=lambda k: Y[k], reverse=True)
        x_p = [x_p[idx] for idx in sort_idx]
        y_p = [y_p[idx] for idx in sort_idx]
        obj_p = parents_obj[0][:10].get("F")
        obj_p = [obj_p[idx] for idx in sort_idx]

        # 构建提示
        prompt_content = self.get_prompt(x_p, y_p, obj_p)

        if debug_mode:
            print("🤖 提示内容:")
            print(prompt_content)
            print("> 按回车继续")
            input()

        # 根据端点类型选择API调用方式
        if "localhost" in endpoint or "127.0.0.1" in endpoint:
            # 本地Ollama
            response = self.call_ollama_api(prompt_content, model_LLM, endpoint)
        else:
            # OpenAI或其他兼容API
            response = self.call_openai_api(prompt_content, model_LLM, endpoint, key)

        if response is None:
            print("❌ 所有API调用失败，返回随机解")
            # 返回随机解作为后备
            off1 = np.random.rand(len(x_p[0]))
            off2 = np.random.rand(len(x_p[0]))
        else:
            # 解析响应
            matches = re.findall(r"<start>(.*?)<end>", response)
            
            if len(matches) >= 2:
                try:
                    off_string1 = matches[0]
                    off1 = np.fromstring(off_string1, sep=",", dtype=float)
                    
                    off_string2 = matches[1]
                    off2 = np.fromstring(off_string2, sep=",", dtype=float)
                    
                    print(f"✅ 成功解析出 {len(matches)} 个解")
                except Exception as e:
                    print(f"❌ 解析响应失败: {str(e)}")
                    off1 = np.random.rand(len(x_p[0]))
                    off2 = np.random.rand(len(x_p[0]))
            else:
                print(f"⚠️  只找到 {len(matches)} 个解，使用随机解补充")
                off1 = np.random.rand(len(x_p[0]))
                off2 = np.random.rand(len(x_p[0]))

        # 保存记录
        if out_filename is not None:
            try:
                with open(out_filename, "a") as file:
                    for i in range(len(x_p)):
                        for j in range(len(x_p[i])):
                            file.write("{:.4f} ".format(x_p[i][j]))
                        file.write("{:.4f} ".format(y_p[i]))
                    for i in range(len(off1)):
                        file.write("{:.4f} ".format(off1[i]))
                    for i in range(len(off1)):
                        file.write("{:.4f} ".format(off2[i]))
                    file.write("\n")
            except Exception as e:
                print(f"⚠️  保存记录失败: {str(e)}")

        # 边界处理
        off1 = np.clip(off1, 0.0, 1.0)
        off2 = np.clip(off2, 0.0, 1.0)
        
        # 反变换
        off1 = np.array([[(off1 * (_.xu - _.xl) + _.xl)]])
        off2 = np.array([[(off2 * (_.xu - _.xl) + _.xl)]])
        off = np.append(off1, off2, axis=0)

        if debug_mode:
            print("🎯 最终结果:")
            print(f"响应: {response}")
            print(f"解1: {off1}")
            print(f"解2: {off2}")
            print("> 按回车继续")
            input()

        return off


class GPT_OpenAI_interface(GPT_OpenAI):
    """GPT OpenAI接口类"""

    def __init__(self, **kwargs):
        super().__init__(n_new=1, **kwargs) 