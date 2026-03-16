# LangGraph 高级智能体

基于 LangGraph 框架和 DeepSeek 模型的高级智能体实现。

## 功能特性

- 🤖 基于 LangGraph 的状态图工作流
- 🧠 集成大语言模型
- 🔧 支持多种工具：搜索、计算、时间查询、终端命令
- 📝 详细的执行过程日志
- 🔄 智能的工具调用管理

## 项目结构

```
langgraph_agent/
├── main.py              # 项目入口
├── agent.py             # 智能体包装类
├── graph.py             # LangGraph 状态图定义
├── deepseek_llm.py      # 模型集成(建议deepseek)
├── tools.py             # 工具定义
├── .env.example         # 环境变量示例
└── README.md            # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

1. 复制环境变量示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的 API 密钥：
```env
DEEPSEEK_API_KEY=your_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # 可选
```

## 使用方法

### 交互式模式
```bash
python main.py
```

### 命令行参数模式
```bash
python main.py "现在是什么时间"
python main.py "12345 + 67890"
python main.py "帮我在桌面创建一个test文件"
```

## 可用工具

- 🔍 **Search** - 搜索网络信息（需要 Tavily API Key）
- 🧮 **Calculator** - 数学计算
- 🕐 **TodayTime** - 获取当前时间
- 💻 **TerminalCommand** - 执行终端命令

## 技术栈

- Python 3.8+
- LangGraph
- LangChain
- DeepSeek API
- Tavily API (可选)

## 许可证

MIT License
