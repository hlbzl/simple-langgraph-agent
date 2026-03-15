from langchain_core.tools import Tool
from tavily import TavilyClient
import os
import subprocess
import sys
import datetime
from dotenv import load_dotenv

load_dotenv()


def search_tool(query: str) -> str:
    """搜索工具"""
    print(f"    ┌─ 搜索工具执行详情")
    print(f"    │  查询内容: {query}")
    
    try:
        # 从环境变量中获取TAVILY_API_KEY
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            error_msg = "搜索失败: 未设置TAVILY_API_KEY环境变量"
            print(f"    │  ❌ {error_msg}")
            print(f"    └─ 工具执行结束")
            return error_msg
        
        print(f"    │  正在连接 Tavily API...")
        # 创建TavilyClient实例
        tavily = TavilyClient(api_key=tavily_api_key)
        
        # 使用Tavily搜索信息
        print(f"    │  正在搜索...")
        response = tavily.search(query, search_depth="basic")
        
        # 处理搜索结果
        if response.get("results"):
            results = response["results"]
            summary = f"搜索结果: 关于 '{query}' 的信息\n\n"
            print(f"    │  ✅ 找到 {len(results)} 个结果")
            
            for i, result in enumerate(results[:3]):  # 只返回前3个结果
                title = result.get("title", "无标题")
                url = result.get("url", "无链接")
                content = result.get("content", "无内容")
                summary += f"{i+1}. {title}\n链接: {url}\n内容: {content[:100]}...\n\n"
                print(f"    │  结果 {i+1}: {title[:50]}...")
            
            print(f"    └─ 工具执行成功，返回 {len(summary)} 字符")
            return summary
        else:
            msg = f"搜索结果: 未找到关于 '{query}' 的信息"
            print(f"    │  ⚠️ {msg}")
            print(f"    └─ 工具执行结束")
            return msg
    except Exception as e:
        error_msg = f"搜索失败: {str(e)}"
        print(f"    │  ❌ {error_msg}")
        print(f"    └─ 工具执行失败")
        return error_msg


def calculator_tool(expression: str) -> str:
    """计算工具"""
    print(f"    ┌─ 计算工具执行详情")
    print(f"    │  表达式: {expression}")
    
    try:
        result = eval(expression)
        result_str = f"计算结果: {result}"
        print(f"    │  ✅ 计算成功: {result}")
        print(f"    └─ 工具执行结束")
        return result_str
    except Exception as e:
        error_msg = f"计算失败: {str(e)}"
        print(f"    │  ❌ {error_msg}")
        print(f"    └─ 工具执行失败")
        return error_msg


def today_time() -> str:
    """获取当前时间工具"""
    print(f"    ┌─ 时间工具执行详情")
    
    try:
        now = datetime.datetime.now()
        result = f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"    │  ✅ 获取成功: {result}")
        print(f"    └─ 工具执行结束")
        return result
    except Exception as e:
        error_msg = f"获取当前时间失败: {str(e)}"
        print(f"    │  ❌ {error_msg}")
        print(f"    └─ 工具执行失败")
        return error_msg


def terminal_command(command: str) -> str:
    """终端命令执行工具"""
    print(f"    ┌─ 终端命令工具执行详情")
    print(f"    │  命令: {command}")
    print(f"    │")
    print(f"    │  {'='*50}")
    print(f"    │  ⚠️  即将执行命令: {command}")
    print(f"    │  {'='*50}")
    
    # 询问用户确认
    confirmation = input("    │  确认执行此命令吗？(输入 'y' 确认，其他任意键取消): ")
    
    if confirmation.lower() == 'y':
        try:
            print(f"    │")
            print(f"    │  正在执行命令...")
            
            # 对于Windows系统，使用系统默认编码
            if sys.platform.startswith('win'):
                # 在Windows中，使用cmd执行命令，并使用GBK编码解码
                result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='gbk', errors='ignore')
            else:
                # 在其他系统中，使用UTF-8编码
                result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            output = ""
            if result.stdout:
                output += f"标准输出:\n{result.stdout}\n"
            if result.stderr:
                output += f"错误输出:\n{result.stderr}\n"
            
            output += f"返回码: {result.returncode}"
            
            print(f"    │  ✅ 命令执行完成")
            print(f"    │  返回码: {result.returncode}")
            print(f"    │  输出长度: {len(output)} 字符")
            print(f"    └─ 工具执行结束")
            return output
        except Exception as e:
            error_msg = f"命令执行失败: {str(e)}"
            print(f"    │  ❌ {error_msg}")
            print(f"    └─ 工具执行失败")
            return error_msg
    else:
        cancel_msg = "命令执行已取消"
        print(f"    │  ⚠️ {cancel_msg}")
        print(f"    └─ 工具执行结束")
        return cancel_msg


def get_tools() -> list[Tool]:
    """获取所有工具列表"""
    tools = [
        Tool(
            name="Search",
            func=search_tool,
            description="用于搜索信息的工具，可以获取最新的网络信息"
        ),
        Tool(
            name="Calculator",
            func=calculator_tool,
            description="用于进行数学计算的工具，支持加减乘除等运算"
        ),
        Tool(
            name="TodayTime",
            func=today_time,
            description="用于获取当前时间的工具，当回答用户的问题需要用到当前最新时间或日期时可调用"
        ),
        Tool(
            name="TerminalCommand",
            func=terminal_command,
            description="用于在系统终端执行命令的工具"
        )
    ]
    return tools
