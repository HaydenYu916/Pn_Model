import numpy as np
import json
import http.client
import re
import time
from pymoo.core.crossover import Crossover

class OllamaGPT(Crossover):
    """兼容Ollama API的GPT交叉算子"""

    def __init__(self, n_new, **kwargs):
        super().__init__(10, 2, **kwargs)
        self.n_new = n_new

    def get_prompt(self, x, y, obj_p):
        """生成优化提示词"""
        pop_content = " "
        for i in range(len(x)):
            pop_content += "point: <start>" + ",".join(str(idx) for idx in x[i].tolist()) + "<end> \n objective 1: " + str(round(obj_p[i][0], 4)) + " objective 2: " + str(round(obj_p[i][1], 4)) + "\n\n"
        
        prompt_content = "Now you will help me minimize " + str(len(obj_p[0])) + " objectives with " + str(len(x[0])) + " variables. I have some points with their objective values. The points start with <start> and end with <end>.\n\n" \
                        + pop_content \
                        + "Give me two new points that are different from all points above, and not dominated by any of the above. Do not write code. Do not give any explanation. Each output new point must start with <start> and end with <end>"
        return prompt_content

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

        # 生成提示词
        prompt_content = self.get_prompt(x_p, y_p, obj_p)

        if debug_mode:
            print(prompt_content)
            print("> enter to continue")
            input()

        # 构建Ollama API请求
        payload = json.dumps({
            "model": model_LLM,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        })
        
        headers = {
            'Content-Type': 'application/json'
        }

        # 连接Ollama API
        import requests
        
        retries = 50
        retry_delay = 2
        while retries > 0:
            try:
                # 使用requests处理流式响应
                response_obj = requests.post(f"http://{endpoint}/api/chat", 
                                           json=json.loads(payload), 
                                           headers=headers, 
                                           timeout=60)
                
                if response_obj.status_code != 200:
                    raise Exception(f"HTTP {response_obj.status_code}")
                
                # 处理流式响应
                response = ""
                for line in response_obj.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'message' in data and 'content' in data['message']:
                                response += data['message']['content']
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue

                # 检查响应格式
                while len(re.findall(r"<start>(.*?)<end>", response)) < 2:
                    response_obj = requests.post(f"http://{endpoint}/api/chat", 
                                               json=json.loads(payload), 
                                               headers=headers, 
                                               timeout=60)
                    response = ""
                    for line in response_obj.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'message' in data and 'content' in data['message']:
                                    response += data['message']['content']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue

                # 解析响应
                off_string1 = re.findall(r"<start>(.*?)<end>", response)[0]
                off1 = np.fromstring(off_string1, sep=",", dtype=float)

                off_string2 = re.findall(r"<start>(.*?)<end>", response)[1]
                off2 = np.fromstring(off_string2, sep=",", dtype=float)

                # 记录输出
                if out_filename is not None:
                    filename = out_filename
                    with open(filename, "a") as file:
                        for i in range(len(x_p)):
                            for j in range(len(x_p[i])):
                                file.write("{:.4f} ".format(x_p[i][j]))
                            file.write("{:.4f} ".format(y_p[i]))
                        for i in range(len(off1)):
                            file.write("{:.4f} ".format(off1[i]))
                        for i in range(len(off1)):
                            file.write("{:.4f} ".format(off2[i]))
                        file.write("\n")

                # 边界处理
                off1[np.where(off1 < 0)] = 0.0
                off1[np.where(off1 > 1)] = 1.0
                off2[np.where(off2 < 0)] = 0.0
                off2[np.where(off2 > 1)] = 1.0
                
                # 反归一化
                off1 = np.array([[(off1 * (_.xu - _.xl) + _.xl)]])
                off2 = np.array([[(off2 * (_.xu - _.xl) + _.xl)]])
                off = np.append(off1, off2, axis=0)

                break

            except Exception as e:
                print(f"Request {retries} failed: {str(e)}")
                retries -= 1
                if retries > 0:
                    print("Retrying in", retry_delay, "seconds...")
                    time.sleep(retry_delay)

        if debug_mode:
            print(response)
            print(off_string1)
            print(off_string2)
            print(off)
            print("> enter to continue")
            input()

        return off


class OllamaGPT_interface(OllamaGPT):
    """Ollama GPT接口类"""
    
    def __init__(self, **kwargs):
        super().__init__(n_new=1, **kwargs) 