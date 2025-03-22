# Customizable Chatbot with LangChain

A versatile conversational AI chatbot built with LangChain, featuring memory and user-selectable themes. This project showcases the development of an interactive AI agent that adapts to different roles—Tech Support, Fitness Coach, or Financial Advisor—while remembering past interactions.

## Features
- **Conversational Memory**: Retains context from previous messages using LangGraph.
- **Selectable Themes**: Choose from:
  - **Tech Support**: Assists with software-related queries.
  - **Fitness Coach**: Offers workout routines, diet plans, and motivation.
  - **Financial Advisor**: Provides insights on finance, investing, and markets.
- **Powered by OpenAI**: Utilizes the `gpt-4o-mini` model for natural language generation.

## Prerequisites
- Python 3.8 or later
- API keys:
  - [LangSmith API Key](https://smith.langchain.com) for tracing
  - [OpenAI API Key](https://platform.openai.com/account/api-keys) for the LLM
- A `.env` file with the following:
