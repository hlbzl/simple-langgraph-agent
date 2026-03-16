#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LangGraph的高级智能体
使用DeepSeek模型和硅基流动平台API
"""

import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from agent import AdvancedAgent


def main():
    """主函数"""
    print("=" * 60)
    print("LangGraph 高级智能体")
    print("基于 DeepSeek 模型和硅基流动平台 API")
    print("=" * 60)
    print()
    
    # 初始化智能体
    agent = AdvancedAgent()
    
    print()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 使用命令行参数作为输入
        query = ' '.join(sys.argv[1:])
        print(f"收到命令行输入: {query}")
        # 运行智能体
        result = agent.run(query)
        # 输出结果
        print(f"\n智能体回答: {result}")
    # 检查是否有管道输入
    elif not sys.stdin.isatty():
        # 处理管道输入
        try:
            # 读取所有输入（二进制模式）
            pipe_input = sys.stdin.buffer.read().strip()
            if pipe_input:
                # 尝试解码为GBK（Windows默认编码）
                try:
                    pipe_input = pipe_input.decode('gbk')
                except UnicodeDecodeError:
                    # 如果GBK解码失败，尝试UTF-8
                    try:
                        pipe_input = pipe_input.decode('utf-8')
                    except UnicodeDecodeError:
                        # 如果都失败，使用replace模式
                        pipe_input = pipe_input.decode('utf-8', errors='replace')
                print(f"收到管道输入: {pipe_input}")
                # 运行智能体
                result = agent.run(pipe_input)
                # 输出结果
                print(f"\n智能体回答: {result}")
            else:
                print("管道输入为空")
        except Exception as e:
            print(f"处理管道输入时发生错误: {str(e)}")
    else:
        # 交互式输入
        print("智能体已启动，输入 'exit' 退出")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n请输入您的问题: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    print("\n感谢使用，再见！")
                    break
                
                # 运行智能体
                result = agent.run(user_input)
                
                # 输出结果
                print(f"\n智能体回答: {result}")
                
            except KeyboardInterrupt:
                print("\n\n感谢使用，再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {str(e)}")


if __name__ == "__main__":
    main()
