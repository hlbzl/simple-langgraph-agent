from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
import re
import time

from deepseek_llm import DeepSeekLLM
from tools import get_tools


class AgentState(TypedDict):
    """定义智能体状态"""
    messages: Annotated[Sequence[BaseMessage], "对话消息列表"]
    next_step: Annotated[str, "下一步操作"]


class LangGraphAgent:
    """基于LangGraph的智能体实现"""
    
    def __init__(self):
        self.model = DeepSeekLLM()
        self.tools = get_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.graph = self._build_graph()
        self.step_count = 0  # 步骤计数器
    
    def _build_graph(self):
        """构建LangGraph状态图"""
        # 定义工作流图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        
        # 设置入口节点
        workflow.set_entry_point("agent")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # 添加工具节点到agent节点的边
        workflow.add_edge("tools", "agent")
        
        # 编译图
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Agent节点：调用模型生成响应"""
        self.step_count += 1
        print(f"\n{'='*60}")
        print(f"🤖 [步骤 {self.step_count}] Agent 节点 - 调用模型生成响应")
        print(f"{'='*60}")
        
        messages = state["messages"]
        
        # 打印当前消息历史
        print(f"\n📜 当前消息历史 ({len(messages)} 条):")
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                print(f"  [{i}] 👤 用户: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            elif isinstance(msg, AIMessage):
                print(f"  [{i}] 🤖 AI: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            elif isinstance(msg, ToolMessage):
                print(f"  [{i}] 🔧 工具({msg.tool_call_id}): {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        
        # 构建系统提示词
        system_prompt = self._build_system_prompt()
        print(f"\n📝 系统提示词长度: {len(system_prompt)} 字符")
        
        # 构建完整的消息列表
        full_messages = [{"role": "system", "content": system_prompt}]
        
        # 转换消息格式
        for msg in messages:
            if isinstance(msg, HumanMessage):
                full_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                full_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                full_messages.append({"role": "user", "content": f"工具执行结果: {msg.content}"})
        
        # 调用模型
        print(f"\n🚀 正在调用 DeepSeek 模型...")
        start_time = time.time()
        response = self.model.invoke(full_messages)
        elapsed_time = time.time() - start_time
        print(f"✅ 模型调用完成，耗时: {elapsed_time:.2f} 秒")
        
        # 打印模型响应
        content = response.content
        print(f"\n💬 模型响应:")
        print(f"{'-'*60}")
        print(content[:500] if len(content) > 500 else content)
        if len(content) > 500:
            print(f"... (共 {len(content)} 字符)")
        print(f"{'-'*60}")
        
        # 检查是否包含工具调用
        has_tool = self._has_tool_call(content)
        print(f"\n🔍 是否包含工具调用: {'是' if has_tool else '否'}")
        
        # 添加AI消息到状态
        new_messages = list(messages) + [response]
        
        return {
            "messages": new_messages,
            "next_step": "continue" if has_tool else "end"
        }
    
    def _tool_node(self, state: AgentState) -> AgentState:
        """工具节点：执行工具调用"""
        self.step_count += 1
        print(f"\n{'='*60}")
        print(f"🔧 [步骤 {self.step_count}] Tools 节点 - 执行工具调用")
        print(f"{'='*60}")
        
        messages = state["messages"]
        last_message = messages[-1]
        
        if not isinstance(last_message, AIMessage):
            print(f"⚠️ 最后一条消息不是 AI 消息，跳过工具执行")
            return state
        
        # 解析工具调用 - 只解析最后一条AI消息中的工具调用
        print(f"\n📋 解析最后一条AI消息中的工具调用...")
        tool_calls = self._parse_tool_calls(last_message.content)
        print(f"✅ 发现 {len(tool_calls)} 个工具调用")
        
        # 检查是否已经执行过这些工具调用（避免重复执行）
        # 获取已执行的工具名称列表
        executed_tools = set()
        for msg in messages[:-1]:  # 不包括最后一条AI消息
            if isinstance(msg, ToolMessage):
                executed_tools.add(msg.tool_call_id)
        
        # 过滤掉已经执行过的工具调用
        new_tool_calls = []
        for tool_name, tool_params in tool_calls:
            if tool_name not in executed_tools:
                new_tool_calls.append((tool_name, tool_params))
            else:
                print(f"⚠️ 工具 {tool_name} 已经执行过，跳过")
        
        tool_calls = new_tool_calls
        print(f"✅ 实际需要执行的工具调用: {len(tool_calls)} 个")
        
        # 如果没有新的工具需要执行，直接返回
        if not tool_calls:
            print(f"\n⚠️ 没有新的工具需要执行，返回当前状态")
            return state
        
        # 执行工具
        tool_messages = []
        for i, (tool_name, tool_params) in enumerate(tool_calls):
            print(f"\n{'─'*60}")
            print(f"🔨 工具调用 {i+1}/{len(tool_calls)}: {tool_name}")
            print(f"{'─'*60}")
            print(f"📥 参数: {json.dumps(tool_params, ensure_ascii=False)}")
            
            if tool_name in self.tool_map:
                try:
                    tool = self.tool_map[tool_name]
                    print(f"🎯 执行工具: {tool_name}")
                    
                    start_time = time.time()
                    # 执行工具
                    if tool_name == "Search":
                        result = tool.func(tool_params.get("query", ""))
                    elif tool_name == "Calculator":
                        result = tool.func(tool_params.get("expression", ""))
                    elif tool_name == "TodayTime":
                        result = tool.func()
                    elif tool_name == "TerminalCommand":
                        result = tool.func(tool_params.get("command", ""))
                    else:
                        result = tool.func(**tool_params)
                    
                    elapsed_time = time.time() - start_time
                    print(f"✅ 工具执行完成，耗时: {elapsed_time:.2f} 秒")
                    print(f"📤 结果长度: {len(result)} 字符")
                    
                    # 使用工具名称+参数作为tool_call_id，确保不同命令被视为不同调用
                    call_id = f"{tool_name}:{json.dumps(tool_params, sort_keys=True)}"
                    tool_messages.append(ToolMessage(content=result, tool_call_id=call_id))
                except Exception as e:
                    error_msg = f"工具执行失败: {str(e)}"
                    print(f"❌ {error_msg}")
                    # 使用工具名称+参数作为tool_call_id，确保不同命令被视为不同调用
                    call_id = f"{tool_name}:{json.dumps(tool_params, sort_keys=True)}"
                    tool_messages.append(ToolMessage(content=error_msg, tool_call_id=call_id))
            else:
                error_msg = f"未找到工具: {tool_name}"
                print(f"❌ {error_msg}")
                # 使用工具名称+参数作为tool_call_id，确保不同命令被视为不同调用
                call_id = f"{tool_name}:{json.dumps(tool_params, sort_keys=True)}"
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=call_id))
        
        # 添加工具消息到状态
        new_messages = list(messages) + tool_messages
        
        print(f"\n✨ 工具节点执行完成，新增 {len(tool_messages)} 条工具消息")
        
        return {
            "messages": new_messages,
            "next_step": "agent"
        }
    
    def _should_continue(self, state: AgentState) -> str:
        """决定下一步操作"""
        messages = state["messages"]
        if not messages:
            print(f"\n🛑 没有消息，结束执行")
            return "end"
        
        last_message = messages[-1]
        
        # 如果最后一条是工具消息，说明工具已执行完，需要让Agent继续处理
        if isinstance(last_message, ToolMessage):
            print(f"\n➡️ 工具执行完成，让Agent继续处理")
            return "continue"
        
        # 如果最后一条是AI消息，检查是否包含工具调用
        if isinstance(last_message, AIMessage):
            # 检查是否包含工具调用
            if self._has_tool_call(last_message.content):
                print(f"\n➡️ 检测到工具调用，继续执行")
                return "continue"
            else:
                print(f"\n🏁 AI生成最终回答，结束执行")
                return "end"
        
        # 其他情况（如HumanMessage），让Agent处理
        print(f"\n➡️ 需要Agent处理，继续执行")
        return "continue"
    
    def _has_tool_call(self, content: str) -> bool:
        """检查内容是否包含工具调用"""
        return "TOOL_CALL:" in content
    
    def _parse_tool_calls(self, content: str) -> list:
        """解析工具调用"""
        tool_calls = []
        
        # 找到所有 TOOL_CALL: 的位置
        import re
        pattern = r'TOOL_CALL:\s*'
        matches = list(re.finditer(pattern, content))
        
        for match in matches:
            start_pos = match.end()
            # 从 TOOL_CALL: 后开始解析 JSON
            json_str = self._extract_json(content[start_pos:])
            if json_str:
                try:
                    tool_call = json.loads(json_str)
                    tool_name = tool_call.get("name")
                    tool_params = tool_call.get("params", {})
                    tool_calls.append((tool_name, tool_params))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取完整的 JSON 对象"""
        text = text.strip()
        if not text.startswith('{'):
            return None
        
        # 使用栈来匹配括号
        stack = []
        end_pos = 0
        
        for i, char in enumerate(text):
            if char == '{':
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:  # 栈为空，说明找到了完整的 JSON
                        end_pos = i + 1
                        break
        
        if end_pos > 0:
            return text[:end_pos]
        return None
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tools_description = "\n\n可用工具:\n"
        for tool in self.tools:
            tools_description += f"- {tool.name}: {tool.description}\n"
        
        tools_description += "\n当你需要使用工具时，请按照以下格式输出:\n"
        tools_description += 'TOOL_CALL: {"name": "工具名称", "params": {"参数名": "参数值"}}\n'
        tools_description += '例如: TOOL_CALL: {"name": "Calculator", "params": {"expression": "1+1"}}\n'
        tools_description += "\n当用户的问题是基于当前时间或日期时，必须先调用TodayTime工具获取最新时间。"
        
        return f"you are a helpful assistant who can use the tools provided to answer the user's questions.Answer the following questions as best you can.{tools_description}"
    
    def run(self, query: str) -> str:
        """运行智能体处理查询"""
        print(f"\n{'#'*60}")
        print(f"# 🚀 开始处理用户查询")
        print(f"# 📝 查询内容: {query}")
        print(f"{'#'*60}")
        
        # 重置步骤计数器
        self.step_count = 0
        
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next_step": "agent"
        }
        
        # 执行图
        print(f"\n📊 开始执行 LangGraph 工作流...")
        start_time = time.time()
        final_state = self.graph.invoke(initial_state)
        total_time = time.time() - start_time
        
        print(f"\n{'#'*60}")
        print(f"# ✅ 工作流执行完成")
        print(f"# ⏱️ 总耗时: {total_time:.2f} 秒")
        print(f"# 📊 总步骤数: {self.step_count}")
        print(f"# 💬 总消息数: {len(final_state['messages'])}")
        print(f"{'#'*60}")
        
        # 提取最终回答
        messages = final_state["messages"]
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                # 移除工具调用标记，只保留回答内容
                content = last_message.content
                if "TOOL_CALL:" in content:
                    # 如果有工具调用，找到最后一个非工具调用的部分
                    parts = content.split("TOOL_CALL:")
                    for part in reversed(parts):
                        part = part.strip()
                        if part and not part.startswith('{'):
                            return part
                return content
        
        return "无法生成回答"
