from graph import LangGraphAgent


class AdvancedAgent:
    """高级智能体 - 基于LangGraph实现"""
    
    def __init__(self):
        """初始化智能体"""
        print("\n" + "="*60)
        print("🚀 初始化 LangGraph 智能体")
        print("="*60)
        
        self.agent = LangGraphAgent()
        
        print("\n✅ 智能体初始化完成")
        print("\n📋 可用工具列表:")
        print("   ┌─ 🔍 Search       - 搜索网络信息")
        print("   ├─ 🧮 Calculator   - 数学计算")
        print("   ├─ 🕐 TodayTime    - 获取当前时间")
        print("   └─ 💻 TerminalCommand - 执行终端命令")
        print("\n" + "="*60)
    
    def run(self, query: str) -> str:
        """运行智能体处理查询"""
        return self.agent.run(query)
