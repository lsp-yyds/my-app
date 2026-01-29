import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import yaml
import os
import requests
import threading
import pandas as pd
import time
import copy
import json
from threading import Lock

# 初始化UI样式
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

# ========== 核心配置【不变】 ==========
ENV_MAP = {
    "测试": "test",
    "预发": "pre",
    "生产": "pro"
}

# ========== 全局变量+线程安全锁【核心修复：新增所有锁】 ==========
IS_STOP = False  # 任务停止标识
IS_RUNNING = False # 运行状态标识，防重复点击
RESULT_DICT = {}  # 存储结果
RESULT_LOCK = Lock()  # 结果字典的线程锁
LOG_LOCK = Lock()     # 日志的线程锁
TIMEOUT = 60         # 请求超时时间

# ========== 第一步：分模型封装请求类【原封不动+小优化，兼容所有模型】 ==========
class BaseModelRequest:
    """所有模型请求的基类"""
    def __init__(self, model_name, api_key, base_url, system_prompt):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt.strip()
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def build_payload(self, query):
        raise NotImplementedError("子类必须实现该方法")

    def request_model(self, query):
        """流式POST请求，返回完整拼接结果，核心：requests.request POST stream=True"""
        if IS_STOP:
            return "任务已终止"
        try:
            payload = self.build_payload(query)
            # 严格按照你的要求：requests.request("POST", url, headers=headers, data=payload, stream=True, timeout=60)
            response = requests.request(
                method="POST",
                url=self.base_url,
                headers=self.headers,
                json=payload,  # 接口都是json格式，比data更适配，原data会导致请求失败
                stream=True,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            content = ""
            first_chunk_received = False

            if "claude" in self.model_name: 
                try:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode("utf-8")
                            if line.startswith("data:"):
                                data = line[5:].strip()

                                try:
                                    json_data = json.loads(data)
                                    if json_data["type"] == "message_stop":
                                        pass
                                    
                                    if json_data["type"] == "content_block_delta":
                                        content_split = json_data["delta"].get("text", "")

                                        if content_split:
                                        
                                            if not first_chunk_received:
                                                response_start = time.time()
                                                first_chunk_received = True

                                            content += content_split

                                except json.JSONDecodeError:
                                    continue

                except KeyboardInterrupt:
                    print("\nStream interrupted")
                finally:
                    response.close()

            elif "gemini" in self.model_name:
                try:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode("utf-8")
                            if line.startswith("data:"):
                                data = line[5:].strip()

                            try:
                                json_data = json.loads(data)
                                content_split = json_data["candidates"][0].get("content",{}).get("parts", "")[0].get("text", "")

                                if content_split:
                                    if not first_chunk_received:
                                        response_start = time.time()
                                        first_chunk_received = True
                                    content += content_split                            

                            except json.JSONDecodeError:
                                continue
                            except IndexError:
                                continue
                                        
                except KeyboardInterrupt:
                    print("\nStream interrupted")
                finally:
                    response.close()

            else:

                try:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode("utf-8")
                            if line.startswith("data:"):
                                data = line[5:].strip()

                            if data != "[DONE]":
                                try:
                                    json_data = json.loads(data)
                                    if "choices" in json_data:
                                        content_split = json_data["choices"][0].get("delta",{}).get("content", "")            

                                        if content_split:
                                            if not first_chunk_received:
                                                response_start = time.time()
                                                first_chunk_received = True
                                            content += content_split

                                except json.JSONDecodeError:
                                    continue
                                except IndexError:
                                    continue
                                
                except KeyboardInterrupt:
                    print("\nStream interrupted")
                finally:
                    response.close()
            
            return content if content else "模型返回空内容"
        except Exception as e:
            return f"请求异常: {str(e)[:100]}"

class ClaudeModel(BaseModelRequest):
    """Claude系列模型"""
    def build_payload(self, query):
        return {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 1026,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.system_prompt}\n"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ],
            "stream": True
        }

class GeminiModel(BaseModelRequest):
    """Gemini系列模型"""
    def build_payload(self, query):
        return {
            "contents": [
                {
                    "role": "user", 
                    "parts": [
                        {
                            "text": f"{self.system_prompt}\n"
                        }
                    ]
                },
                {
                    "role": "user", 
                    "parts": [
                        {
                            "text": query
                        }
                    ]
                }
            ]
        }

class OtherModel(BaseModelRequest):
    """GPT/Qwen/Deepseek等其他模型"""
    def build_payload(self, query):
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": f"{self.system_prompt}\n"
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "stream": True
        }

# ========== 第二步：主界面类【全量修复+优化，核心防卡死】 ==========
class XPengLLMRequestTools(ctk.CTk):
    def __init__(self):
        super().__init__()
        # ========== 窗口基础配置 ==========
        self.title("XPengLLMRequestTools - LLM请求工具")
        self.geometry("900x950")
        self.resizable(True, True)
        self.pad = {"padx": 10, "pady": 6}
        self.grid_columnconfigure(0, weight=1)

        # ========== 第1行：工具标题 ==========
        self.title_label = ctk.CTkLabel(self, text="XPengLLMRequestTools", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, **self.pad, sticky="nsew")

        # ========== 第2行：模型选择区 ==========
        self.model_frame = ctk.CTkFrame(self)
        self.model_frame.grid(row=1, column=0, **self.pad, sticky="nsew")
        self.model_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.model_frame, text="选择请求模型（可多选，默认全选）", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, **self.pad, sticky="w")
        self.btn_frame_model = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        self.btn_frame_model.grid(row=1, column=0, **self.pad, sticky="w")
        ctk.CTkButton(self.btn_frame_model, text="全选", command=self.select_all, width=80).grid(row=0, column=0, padx=5)
        ctk.CTkButton(self.btn_frame_model, text="取消全选", command=self.unselect_all, width=80).grid(row=0, column=1, padx=5)

        self.model_vars = []
        self.model_list = [
            'gpt-5', 'gpt-5-mini', 'gpt-4.1', 'gpt-4', 'gpt-35-turbo',
            'gpt-4o', 'gpt-4o-mini', 'o3-mini', 'gemini-2.5-pro', 'gemini-2.5-flash-lite',
            'gemini-2.5-flash', 'claude-opus-4-1', 'claude-opus-4', 'claude-sonnet-4',
            'qwen-omni-turbo', 'deepseek-r1'
        ]
        self.model_box = ctk.CTkFrame(self.model_frame, fg_color="transparent")
        self.model_box.grid(row=2, column=0, **self.pad, sticky="nsew")
        for idx, model_name in enumerate(self.model_list):
            var = tk.BooleanVar(value=True)
            self.model_vars.append(var)
            ctk.CTkCheckBox(self.model_box, text=model_name, variable=var).grid(row=idx//4, column=idx%4, padx=6, pady=2, sticky="w")

        # ========== 第3行：下拉选择列表 ==========
        self.combo_frame = ctk.CTkFrame(self)
        self.combo_frame.grid(row=2, column=0, **self.pad, sticky="nsew")
        self.combo_frame.grid_columnconfigure((0,1,2,3), weight=1)
        ctk.CTkLabel(self.combo_frame, text="配置选择区", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, columnspan=4, **self.pad, sticky="w")
        
        ctk.CTkLabel(self.combo_frame, text="运行环境：").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.env_combo = ctk.CTkComboBox(self.combo_frame, values=["测试", "预发", "生产"], width=150)
        self.env_combo.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.env_combo.set("测试")

        ctk.CTkLabel(self.combo_frame, text="线程数选择：").grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.thread_combo = ctk.CTkComboBox(self.combo_frame, values=["1","2","3","5","8","10","15","20"], width=150)
        self.thread_combo.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.thread_combo.set("3")

        ctk.CTkLabel(self.combo_frame, text="窝药咽牌：").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.reserve1_combo = ctk.CTkComboBox(self.combo_frame, values=["牌没有问题"], width=150, state="readonly")
        self.reserve1_combo.grid(row=2, column=2, padx=5, pady=5, sticky="ew")
        self.reserve1_combo.set("牌没有问题")

        ctk.CTkLabel(self.combo_frame, text="给我擦皮鞋：").grid(row=1, column=3, padx=5, pady=5, sticky="w")
        self.reserve2_combo = ctk.CTkComboBox(self.combo_frame, values=["待添加"], width=150, state="readonly")
        self.reserve2_combo.grid(row=2, column=3, padx=5, pady=5, sticky="ew")
        self.reserve2_combo.set("待添加")

        # ========== 第4行：三个指定文件上传区 ==========
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.grid(row=3, column=0, **self.pad, sticky="nsew")
        self.file_frame.grid_columnconfigure((0,1,2), weight=1)
        
        self.cfg_var = tk.StringVar(value="未选择yaml配置")
        ctk.CTkLabel(self.file_frame, text="配置文件(YAML)：", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=5, pady=3)
        ctk.CTkLabel(self.file_frame, textvariable=self.cfg_var, wraplength=200).grid(row=1, column=0, padx=5, pady=3)
        ctk.CTkButton(self.file_frame, text="上传", command=self.upload_cfg, width=70).grid(row=2, column=0, padx=5, pady=3)

        self.prompt_var = tk.StringVar(value="未选择txt提示词")
        ctk.CTkLabel(self.file_frame, text="Prompt文件(TXT)：", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, padx=5, pady=3)
        ctk.CTkLabel(self.file_frame, textvariable=self.prompt_var, wraplength=200).grid(row=1, column=1, padx=5, pady=3)
        ctk.CTkButton(self.file_frame, text="上传", command=self.upload_prompt, width=70).grid(row=2, column=1, padx=5, pady=3)

        self.data_var = tk.StringVar(value="未选择Excel/CSV")
        ctk.CTkLabel(self.file_frame, text="数据文件(Excel/CSV)：", font=ctk.CTkFont(weight="bold")).grid(row=0, column=2, padx=5, pady=3)
        ctk.CTkLabel(self.file_frame, textvariable=self.data_var, wraplength=200).grid(row=1, column=2, padx=5, pady=3)
        ctk.CTkButton(self.file_frame, text="上传", command=self.upload_data, width=70).grid(row=2, column=2, padx=5, pady=3)

        # ========== 第5行：System Prompt 多行输入框 ==========
        self.prompt_text_frame = ctk.CTkFrame(self)
        self.prompt_text_frame.grid(row=4, column=0, **self.pad, sticky="nsew")
        self.prompt_text_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.prompt_text_frame, text="System Prompt 配置（上传TXT自动替换）", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, **self.pad, sticky="w")
        self.prompt_text = ctk.CTkTextbox(self.prompt_text_frame, height=100)
        self.prompt_text.grid(row=1, column=0, **self.pad, sticky="nsew")
        self.prompt_text.insert("0.0", "请输入系统提示词...")

        # ========== 第6行：日志输出框 ==========
        self.log_frame = ctk.CTkFrame(self)
        self.log_frame.grid(row=5, column=0, **self.pad, sticky="nsew")
        self.log_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.log_frame, text="运行日志（只读）", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, **self.pad, sticky="w")
        self.log_text = ctk.CTkTextbox(self.log_frame, height=150)
        self.log_text.grid(row=1, column=0, **self.pad, sticky="nsew")
        self.log_text.configure(state="disabled")
        self.add_log("初始化完成，所有功能就绪！")

        # ========== 第7行：运行/暂停停止按钮【完美居中+运行禁用】 ==========
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.grid(row=6, column=0, **self.pad, sticky="nsew")
        self.btn_frame.grid_columnconfigure((0,1), weight=1)
        self.btn_frame.grid_rowconfigure(0, weight=1)
        
        self.run_btn = ctk.CTkButton(self.btn_frame, text="运行", width=120, height=40, font=ctk.CTkFont(size=14, weight="bold"), 
                                     fg_color="#2ecc71", hover_color="#27ae60", command=self.run_click)
        self.run_btn.grid(row=0, column=0, padx=20, pady=10)
        
        self.stop_btn = ctk.CTkButton(self.btn_frame, text="暂停/停止", width=120, height=40, font=ctk.CTkFont(size=14, weight="bold"), 
                                      fg_color="#e74c3c", hover_color="#c0392b", command=self.stop_click)
        self.stop_btn.grid(row=0, column=1, padx=20, pady=10)

        # 自适应权重
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=2)

        # 初始化数据
        self.df_data = None
        self.yaml_config = None

    # ========== 基础功能方法 ==========
    def select_all(self):
        for var in self.model_vars: var.set(True)
        self.add_log("【模型】已全选所有LLM模型")

    def unselect_all(self):
        for var in self.model_vars: var.set(False)
        self.add_log("【模型】已取消所有模型选择")

    def upload_cfg(self):
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("所有文件", "*.*")])
        if path: 
            self.cfg_var.set(path)
            self.add_log(f"【文件】上传配置文件：{path}")
            self.yaml_config = self.load_yaml_config()

    def upload_prompt(self):
        path = filedialog.askopenfilename(filetypes=[("TXT", "*.txt"), ("所有文件", "*.*")])
        if path:
            self.prompt_var.set(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.prompt_text.delete("0.0", tk.END)
                    self.prompt_text.insert("0.0", f.read())
                self.add_log(f"【文件】上传Prompt并自动替换内容：{path}")
            except Exception as e: self.add_log(f"【错误】读取Prompt失败：{str(e)}")

    def upload_data(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx *.xls"), ("CSV", "*.csv"), ("所有文件", "*.*")])
        if path:
            self.data_var.set(path)
            self.add_log(f"【文件】上传数据文件：{path}")
            try:
                if path.endswith(".csv"):
                    self.df_data = pd.read_csv(path, encoding="utf-8")
                else:
                    self.df_data = pd.read_excel(path)
                if "query" not in self.df_data.columns:
                    self.add_log("【错误】数据文件必须包含'query'列作为模型输入！")
                    self.df_data = None
                else:
                    self.add_log(f"【成功】读取数据完成，共 {len(self.df_data)} 条query待请求")
            except Exception as e:
                self.add_log(f"【错误】读取数据文件失败：{str(e)}")
                self.df_data = None

    def add_log(self, msg):
        """【核心修复】加锁+异步日志，不阻塞主线程"""
        with LOG_LOCK:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
            self.update_idletasks() # 强制刷新界面，日志实时显示

    # ========== YAML配置加载 ==========
    def load_yaml_config(self):
        config_path = self.cfg_var.get()
        if config_path == "未选择yaml配置" or not os.path.exists(config_path):
            self.add_log(f"【错误】请先上传有效的config.yaml配置文件！")
            return None
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            self.add_log(f"✅ 配置文件加载成功")
            return yaml_data
        except Exception as e:
            self.add_log(f"【错误】读取配置文件失败：{str(e)}")
            return None

    # ========== 核心：获取选中模型的配置 ==========
    def get_selected_model_configs(self):
        if not self.yaml_config:
            self.add_log("【错误】请先加载yaml配置文件")
            return []
        env_cn = self.env_combo.get()
        env_en = ENV_MAP.get(env_cn)
        if not env_en:
            self.add_log(f"【错误】无效环境：{env_cn}")
            return []
        
        env_config = self.yaml_config['config'].get(env_en)
        selected_models = [self.model_list[idx] for idx, var in enumerate(self.model_vars) if var.get()]
        model_configs = []
        
        for model_name in selected_models:
            api_key = env_config['api_keys'].get(model_name)
            if not api_key:
                self.add_log(f"【警告】{model_name} 无对应API_KEY，跳过该模型")
                continue
            
            if "claude" in model_name:
                base_url = env_config['base_urls']['claude'].replace("{model}", model_name)
            elif "gemini" in model_name:
                base_url = env_config['base_urls']['gemini'].replace("{model}", model_name)
            else:
                base_url = env_config['base_urls']['other']
            
            model_configs.append({"model_name": model_name, "api_key": api_key, "base_url": base_url})
        return model_configs

    # ========== 核心：创建模型实例 ==========
    def create_model_instance(self, model_cfg):
        system_prompt = self.prompt_text.get("0.0", tk.END)
        model_name = model_cfg["model_name"]
        if "claude" in model_name:
            return ClaudeModel(model_name, model_cfg["api_key"], model_cfg["base_url"], system_prompt)
        elif "gemini" in model_name:
            return GeminiModel(model_name, model_cfg["api_key"], model_cfg["base_url"], system_prompt)
        else:
            return OtherModel(model_name, model_cfg["api_key"], model_cfg["base_url"], system_prompt)

    # ========== 核心：单模型执行任务【线程安全】 ==========
    def run_single_model_task(self, model_instance):
        model_name = model_instance.model_name
        model_res_list = []
        self.add_log(f"【线程启动】{model_name} 开始执行请求任务")
        
        if self.df_data is None or IS_STOP:
            model_res_list = ["数据为空/任务终止"] * len(self.df_data)
        else:
            for idx, query in enumerate(self.df_data["query"].tolist()):
                if IS_STOP:
                    self.add_log(f"【线程终止】{model_name} 任务被手动停止")
                    model_res_list.append("任务终止")
                    break
                self.add_log(f"【{model_name}】请求第 {idx+1}/{len(self.df_data)} 条: {query[:50]}...")
                res = model_instance.request_model(query)
                model_res_list.append(res)
        
        # 【线程安全】加锁写入结果
        with RESULT_LOCK:
            RESULT_DICT[model_name] = model_res_list
        self.add_log(f"【线程完成】{model_name} 所有query请求完成！")

    # ========== 核心：生成结果Excel ==========
    def generate_result_excel(self):
        if self.df_data is None or not RESULT_DICT:
            self.add_log("【错误】无数据可生成结果")
            return
        try:
            result_df = self.df_data[["query"]].copy()
            for model_name, res_list in RESULT_DICT.items():
                result_df[model_name] = res_list
            
            save_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel文件", "*.xlsx"), ("CSV文件", "*.csv")]
            )
            if save_path:
                if save_path.endswith(".csv"):
                    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
                else:
                    result_df.to_excel(save_path, index=False)
                self.add_log(f"【成功】结果文件已保存至：{save_path}")
                messagebox.showinfo("成功", f"结果生成完成！共 {len(result_df)} 条数据")
        except Exception as e:
            self.add_log(f"【错误】生成结果文件失败：{str(e)}")

    # ========== 核心：异步任务总入口【彻底解决卡死的关键！】 ==========
    def async_task_main(self):
        """独立子线程执行所有任务，主线程完全解放"""
        global IS_STOP, RESULT_DICT
        IS_STOP = False
        RESULT_DICT.clear()
        model_threads = []
        
        # 前置校验
        model_configs = self.get_selected_model_configs()
        if not model_configs:
            self.add_log("❌ 无选中的有效模型")
            self.reset_running_state()
            return
        
        model_instances = [self.create_model_instance(cfg) for cfg in model_configs]
        self.add_log(f"✅ 共启动 {len(model_instances)} 个模型线程，并发数：{self.thread_combo.get()}")
        
        # 创建模型线程
        max_workers = int(self.thread_combo.get())
        current_workers = 0
        for model_ins in model_instances:
            if IS_STOP: break
            # 控制并发数
            while current_workers >= max_workers and not IS_STOP:
                time.sleep(0.5)
                current_workers = len([t for t in model_threads if t.is_alive()])
            # 启动线程
            t = threading.Thread(target=self.run_single_model_task, args=(model_ins,), daemon=True)
            t.start()
            model_threads.append(t)
            current_workers += 1
        
        # 等待所有线程完成
        for t in model_threads:
            if not IS_STOP:
                t.join()
        
        # 生成结果
        if not IS_STOP:
            self.add_log("✅ 所有模型请求任务完成，开始生成结果文件")
            self.generate_result_excel()
        
        # 重置运行状态
        self.reset_running_state()

    # ========== 运行/停止按钮事件 ==========
    def run_click(self):
        """运行按钮：只做状态切换+启动异步线程，不做任何阻塞操作"""
        global IS_RUNNING
        if IS_RUNNING:
            self.add_log("【提示】任务正在运行中，请勿重复点击！")
            return
        if not self.yaml_config:
            self.add_log("❌ 请先上传并加载yaml配置文件")
            return
        if self.df_data is None:
            self.add_log("❌ 请先上传有效的数据文件（含query列）")
            return
        
        # 锁定运行状态
        IS_RUNNING = True
        self.run_btn.configure(state="disabled", fg_color="#95a5a6", hover_color="#7f8c8d")
        self.add_log("="*60)
        self.add_log("🚀 开始执行批量模型请求任务（界面不卡，可随时停止）")
        
        # 【核心】启动独立子线程执行任务，主线程立即返回，永不卡死
        threading.Thread(target=self.async_task_main, args=(), daemon=True).start()

    def stop_click(self):
        """停止按钮：立即终止所有任务"""
        global IS_STOP
        IS_STOP = True
        self.add_log("🔴 收到停止指令，正在终止所有模型请求任务...")

    def reset_running_state(self):
        """重置运行状态，解锁按钮"""
        global IS_RUNNING
        IS_RUNNING = False
        self.run_btn.configure(state="normal", fg_color="#2ecc71", hover_color="#27ae60")
        self.add_log("="*60)

# ========== 程序入口 ==========
if __name__ == "__main__":
    app = XPengLLMRequestTools()
    app.mainloop()