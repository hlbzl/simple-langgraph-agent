from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class DeepSeekLLM(Runnable):
    def __init__(self, model=None, api_key=None, base_url=None):
        self.model = model or os.getenv("DEEPSEEK_MODEL")
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")
    
    def __call__(self, messages, stop=None):
        """支持直接调用"""
        return self.invoke(messages, stop)
    
    def invoke(self, input, config=None, stop=None):
        """处理输入，可能是字符串或消息列表"""
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            # 处理消息列表
            messages = []
            for msg in input:
                try:
                    # 尝试获取role和content属性，适用于各种消息对象
                    if hasattr(msg, "content"):
                        content = getattr(msg, "content")
                        # 确定role
                        if hasattr(msg, "role"):
                            role = getattr(msg, "role")
                        elif "system" in str(type(msg)).lower():
                            role = "system"
                        elif "human" in str(type(msg)).lower():
                            role = "user"
                        elif "ai" in str(type(msg)).lower():
                            role = "assistant"
                        else:
                            role = "user"
                        messages.append({"role": role, "content": content})
                    elif isinstance(msg, dict):
                        # 处理字典格式的消息
                        if "role" in msg and "content" in msg:
                            messages.append({"role": msg["role"], "content": msg["content"]})
                except Exception:
                    pass
        else:
            raise ValueError(f"不支持的输入类型: {type(input)}")
        
        if not messages:
            raise ValueError("消息列表不能为空")
        
        response = self._chat_completion(messages, stop)
        
        # 返回AIMessage对象，符合LangChain的期望
        return AIMessage(content=response)
    
    def bind(self, **kwargs):
        """实现bind方法，符合Runnable接口"""
        # 创建一个新的DeepSeekLLM实例，绑定额外的参数
        bound_instance = DeepSeekLLM(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url
        )
        # 存储绑定的参数
        bound_instance.bound_kwargs = kwargs
        return bound_instance
    
    def bind_tools(self, tools, **kwargs):
        """实现bind_tools方法，用于绑定工具信息"""
        print(f"=== bind_tools被调用 ===")
        print(f"工具数量: {len(tools)}")
        for i, tool in enumerate(tools):
            print(f"工具 {i}: {tool.name}, 描述: {tool.description}")
        print(f"绑定的参数: {kwargs}")
        # 创建一个新的DeepSeekLLM实例，绑定工具信息
        bound_instance = DeepSeekLLM(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url
        )
        # 存储绑定的工具和参数
        bound_instance.bound_tools = tools
        bound_instance.bound_kwargs = kwargs
        print(f"=== bind_tools调用结束 ===")
        return bound_instance
    
    def _chat_completion(self, messages, stop=None):
        """调用硅基流动平台的API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # 构建请求数据
        data = {
            "model": self.model,
            "messages": messages
        }
        # 只在stop不为None时添加stop参数
        if stop is not None:
            data["stop"] = stop
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")
