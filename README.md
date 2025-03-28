# create-ai-customer-support-agent-langgraph

- Created at: 2025-03-26
- Created by: `🐢 Arun Godwin Patel @ Code Creations`

## Table of contents

- [Setup](#setup)
  - [System](#system)
  - [Installation](#installation)
- [Walkthrough](#walkthrough)
  - [Code Structure](#code-structure)
  - [Tech stack](#tech-stack)
  - [Build from scratch](#build-from-scratch)
    - [1. Create a virtual environment](#1-create-a-virtual-environment)
    - [2. Activate the virtual environment](#2-activate-the-virtual-environment)
    - [3. Install the required packages](#3-install-the-required-packages)
    - [4. Setup the prompt structure](#4-setup-the-prompt-structure)
    - [5. Create the State class](#5-create-the-state-class)
    - [6. Create the Tools](#6-create-the-tools)
    - [7. Create the AI Agent class](#7-create-the-ai-agent-class)
    - [8. Set the starting conversation state](#8-set-the-starting-conversation-state)
    - [9. Write the main application](#9-write-the-main-application)

## Setup

### System

This code repository was tested on the following computers:

- Windows 11

At the time of creation, this code was built using `Python 3.13.2`

### Installation

1. Install `virtualenv`

```bash
# 1. Open a CMD terminal
# 2. Install virtualenv globally
pip install virtualenv
```

2. Create a virtual environment

```bash
python -m venv venv
```

3. Activate the virtual environment

```bash
# Windows
.\venv\Scripts\activate
# Mac
source venv/bin/activate
```

4. Install the required packages

```bash
pip install -r requirements.txt
```

5. Run the module

```bash
python main.py
```

## Walkthrough

### Code Structure

The code directory structure is as follows:

```plaintext
create-ai-customer-support-agent-langgraph
└───ai
|   └──graph.py
└───data
|   └──customer_inquiry.json
│   .env
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
```

The `main.py` file is the entry point of the application. It imports code from the `ai/graph` file.

The `ai/graph.py` file contains the `State` & `AiAgent` class, which are responsible for defining the state of the conversation, in other words "memory" and the LangGraph AI Agent.

The `data/` folder contains an example JSON file from a customer inquiry that will be utilised by a tool to gather information about the customer that has been contacted..

The `.env` file contains the environment variables used by the application.

The `.gitignore` file specifies the files and directories that should be ignored by Git.

The `requirements.txt` file lists the Python packages required by the application.

### Tech stack

**AI**

- LLM: `Anthopic Claude`

**Orchestration**

- AI Agent: `LangGraph`

### Build from scratch

This project was built from scratch using Python, LangGraph and the Anthopic Claude LLM. The conversational voice agent is contacting a prospective client and has some follow up items that it needs to collect. The agent is equipped with contextual data about the company and their products. It is also armed with a set of tools that provides it with extra functionality.

#### 1. Create a virtual environment

```bash
python -m venv venv
```

#### 2. Activate the virtual environment

```bash
# Windows
.\venv\Scripts\activate
# Mac
source venv/bin/activate
```

#### 3. Install the required packages

```bash
pip install -r requirements.txt
```

#### 4. Setup the prompt structure

In the `ai/graph.py` file we first import the required packages and then we define our static prompts to be used later. These prompts are used to guide the conversation flow between the user and the AI agent.

```python
import json
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

# Constants
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from typing import Literal, Annotated
from typing_extensions import TypedDict

# Constants
MODEL = "claude-3-5-haiku-20241022"
ROLE = (
    "You are a customer support agent for a Domestic Robotics company called HomeBots. "
    "You are following up with prospective customers that have inquired for more information online. "
    "Your goal is to answer their queries and guide them through the process of purchasing a HomeBot. "
)
CONTEXT = (
    "Here is a mandatory checklist of requirements from the customer in order for them to purchase a HomeBot: "
    "\n- Customer's name"
    "\n- Product"
    "\n- Address for delivery"
    "\n- Email address"
    "\n- Phone number"
    "\n- Payment method"
    "\n- Desired delivery date"
    "\n\nAnd here is a product list from HomeBots: "
    "\n- HomeBot 1000"
    "\n- HomeBot 2000"
    "\n- HomeBot 3000"
)
TASK = (
    "1. Greet the customer and confirm their name. If the customer's name is already known in the conversation state, do not ask for it again."
    "\n2. Check the customer inquiry data once (unless new info arises) to see if there are any outstanding items needed from the checklist."
    "\n3. If any checklist items are missing, one by one, ask the customer to provide each of them and save their answer."
    "\n4. If all checklist items are saved, inform the customer that they are ready to purchase a HomeBot."
    "\n4. End the conversation."
)
RULES = (
    "- IMPORTANT: Once you have called the check_required_checklist_items tool once, do not call it again unless the user explicitly asks you to re-check. If state[\"checked_checklist\"] is True, do not call check_required_checklist_items. Instead, continue the conversation without repeating the checklist requirements."
    "- IMPORTANT: Once the checklist items are known in the state, do not ask for them again. If the customer provides new information, update the state accordingly."
    "\n- Use the customer's name if you have it. If the customer's name is already known in the conversation state, do not ask for it again."
    "\n- Always be polite and professional."
    "\n- Be clear and concise in your communication."
    "\n- Only focus on 1 question at a time, do not overwhelm the customer with multiple questions at once."
    "\n- Your goal is to collect all the items in the checklist. It may occur that the customer has questions or needs assistance, be prepared to address those too.The customer may provide answers to the checklist items in an unexpected order, this is fine, just make sure you have all the required information."
    "\n- If the customer has all the required information, inform them that they are ready to purchase a HomeBot."
)
```

#### 5. Create the State class

Next, we define the `State` class that will be used to store the conversation state. The state will be used to keep track of the conversation flow and the information collected from the user.

```python
# Define the State class
class State(TypedDict):
    messages: Annotated[list, add_messages]
    customer_name: str
    product: str
    delivery_address: str
    email_address: str
    phone_number: str
    payment_method: str
    delivery_date: str
    is_finished: bool
    checked_checklist: bool
```

#### 6. Create the Tools

We then define the tools that will be used by the AI agent to interact with the user. These tools will help the AI agent gather information from the user and guide the conversation flow.

```python
# Tools
@tool
def check_required_checklist_items(state: State, tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """
    This tool checks the customer inquiry to see which checklist items are currently available.
    If we've already run this tool before we do a short-circuit response to avoid repeating the same logic.
    """

    # If this tool has been run before, short-circuit
    if state["checked_checklist"]:
        return Command(update={
            "messages": [
                ToolMessage(
                    "We've already checked your details. Please proceed.",
                    tool_call_id=tool_call_id
                )
            ],
            "checked_checklist": True,
            "customer_name": state.get("customer_name", ""),
            "product": state.get("product", ""),
            "delivery_address": state.get("delivery_address", ""),
            "email_address": state.get("email_address", ""),
            "phone_number": state.get("phone_number", ""),
            "payment_method": state.get("payment_method", ""),
            "delivery_date": state.get("delivery_date", ""),
            "is_finished": state.get("is_finished", "")
        })

    # Otherwise, load inquiry data
    inquiry = {}
    with open("data/customer_inquiry.json", "r") as f:
        inquiry = json.load(f)

    # Updaate state using inquiry data
    for key, val in inquiry.items():
        if key in state:
            state[key] = val

    # Check which checklislt items are missing
    requirements = [
        "customer_name",
        "product",
        "delivery_address",
        "email_address",
        "phone_number",
        "payment_method",
        "delivery_date",
    ]
    missing = [item for item in requirements if not state.get(item)]

    # Update state based on required checklist items
    if missing:
        response = f"We need the following checklist items from you before you can purchase a HomeBot: {', '.join(missing)}"
    else:
        response = "All required checklist items have been provided."
        state["is_finished"] = True

    return Command(update={
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
        "checked_checklist": True,
        "customer_name": state.get("customer_name", ""),
        "product": state.get("product", ""),
        "delivery_address": state.get("delivery_address", ""),
        "email_address": state.get("email_address", ""),
        "phone_number": state.get("phone_number", ""),
        "payment_method": state.get("payment_method", ""),
        "delivery_date": state.get("delivery_date", ""),
        "is_finished": state.get("is_finished", "")
    })

@tool
def explain_product_tool(
    state: State,
    tool_call_id: Annotated[str, InjectedToolCallId],
    product: Literal["HomeBot 1000", "HomeBot 2000", "HomeBot 3000"],
) -> str:
    """
    This tool is used to explain a specific product to the customer.
    """

    # Construct a short explanation.
    product_explanations = {
        "HomeBot 1000": "The basic model, does hoovering, washing up and basic cooking. £500",
        "HomeBot 2000": "The advanced model, does hoovering, washing up, cooking, gardening and childcare. £1000",
        "HomeBot 3000": "The deluxe model, does everything the 2000 model does but also has a built-in sauna. £2000",
    }

    explanation = product_explanations.get(
        product,
        "I'm not sure how to explain that product. Please provide a valid product name."
    )

    # Return a response via Command with ToolMessage
    state_update = {
        "messages": [
            ToolMessage(
                content=(
                    f"Here’s the explanation for the {product}:\n\n{explanation}"
                ),
                tool_call_id=tool_call_id,
            )
        ],
        "customer_name": state.get("customer_name", ""),
        "product": state.get("product", ""),
        "delivery_address": state.get("delivery_address", ""),
        "email_address": state.get("email_address", ""),
        "phone_number": state.get("phone_number", ""),
        "payment_method": state.get("payment_method", ""),
        "delivery_date": state.get("delivery_date", ""),
        "is_finished": False,
        "checked_checklist": state.get("checked_checklist", False),
    }
    return Command(update=state_update)

@tool
def parse_date_tool(
    state: State,
    tool_call_id: Annotated[str, InjectedToolCallId],
    iso_datetime: str
) -> str:
    """
    Tool that expects the LLM to provide a date/time
    in 'YYYY-MM-DD HH:MM' format (24-hour clock).

    For instance, if the user says:
      "Let's do a call on March 29th at 5pm"
    the LLM itself must interpret that as
      "2025-03-29 17:00"
    and then call parse_date_tool(date_str="2025-03-29 17:00").
    """

    # Save this parsed date/time in state
    state_update = {
        "messages": [
            ToolMessage(
                content=f"Great, I'll set the delivery date as {iso_datetime}.",
                tool_call_id=tool_call_id,
            )
        ],
        "customer_name": state.get("customer_name", ""),
        "product": state.get("product", ""),
        "delivery_address": state.get("delivery_address", ""),
        "email_address": state.get("email_address", ""),
        "phone_number": state.get("phone_number", ""),
        "payment_method": state.get("payment_method", ""),
        "delivery_date": iso_datetime,
        "is_finished": state.get("is_finished", "")
    }
    return Command(update=state_update)
```

#### 7. Create the AI Agent class

Next, we define the `AiAgent` class that will be used to manage the conversation flow between the user and the AI agent. The `AiAgent` class will be responsible for handling the conversation state and executing the conversation tasks. It is within this class that we define the LangGraph structure including nodes, edges & conditional edges.

```python
# Define the AI Agent
class AiAgent:
    def __init__(self):
        self.model = MODEL
        self.role = ROLE
        self.context = CONTEXT
        self.task = TASK
        self.rules = RULES
        self.llm_with_tools = None

    def build(self):

        # Create the Graph
        graph_builder = StateGraph(State)

        # Instantiate the LLM
        llm = ChatAnthropic(model=MODEL)

        # Bind the tools to the LLM
        tools = [check_required_checklist_items, explain_product_tool, parse_date_tool]
        llm_with_tools = llm.bind_tools(tools)
        self.llm_with_tools = llm_with_tools

        # Add nodes
        graph_builder.add_node("chatbot", self._node_chatbot)
        graph_builder.add_node("end", self._node_end)

        # Add tools nodes
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)

        # Define Edges
        graph_builder.set_entry_point("chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "end",
            },
        )
        graph_builder.add_edge("tools", "chatbot")
        graph = graph_builder.compile()
        return graph

    # Nodes
    def _node_chatbot(self, state: State):
        """
        The LLM sees the conversation so far and can respond or call tools.
        """

        # If customer name exists add a system message to avoid asking for it again
        if state["customer_name"]:
            if all("We already know the customer's name is" not in m.content
                for m in state["messages"] if m.__class__.__name__ == "SystemMessage"):
                override_msg = {
                    "role": "system",
                    "content": (
                        "We already know the customer's name is "
                        f"{state['customer_name']}. Do NOT ask for it again."
                    )
                }
                state["messages"].insert(1, override_msg)

        # The chain can return either a single BaseMessage or a list[BaseMessage].
        # We want to keep them all, so let's standardize to a list:
        responses = self.llm_with_tools.invoke(state["messages"])
        if not isinstance(responses, list):
            responses = [responses]
        return {"messages": responses}

    def _node_end(self, state: State):
        state["is_finished"] = True
        return {"messages": []}
```

#### 8. Set the starting conversation state

We then define the starting conversation state which will be updated throughout the conversation.

```python
CONVERSATION_STATE: State = {
    "messages": [
        {
            "role": "system",
            "content": f"## ROLE\n{ROLE}\n\n## CONTEXT\n{CONTEXT}\n\n## TASK\n{TASK}\n\n## RULES\n{RULES}"
        },
        {
            "role": "assistant",
            "content": (
                "Hello! I'm Hamish from HomeBots! I'm reaching out with regards to your recent product inquiry. "
                "To begin, please confirm your name?"
            ),
        },
    ],
    "customer_name": "",
    "product": "",
    "delivery_address": "",
    "email_address": "",
    "phone_number": "",
    "payment_method": "",
    "delivery_date": "",
    "is_finished": False,
    "checked_checklist": False,
}
```

#### 9. Write the main application

Finally, we write the main application that will instantiate the `AiAgent` class and start the conversation flow between the user and the AI agent.

```python
import json
from ai.graph import State, AiAgent, CONVERSATION_STATE

def main():
    ai_agent = AiAgent()
    graph = ai_agent.build()

    def stream_graph_updates(state: State):
        for event in graph.stream(state):
            for value in event.values():
                for k, v in value.items():
                    if k == "messages":
                        for msg in v:

                            # Store all messages in state
                            state["messages"].append(msg)

                            # Print messages based on type
                            if msg.__class__.__name__ == "ToolMessage":
                                print(f"\t---> 🛠️  Tool: {msg.content}")
                            elif msg.__class__.__name__ == "AIMessage":
                                if type(msg.content) == list:
                                    continue
                                print(f"\n🤖 AI: {msg.content}")

                    # Update state based on key-value pairs
                    elif isinstance(v, list):
                        state[k] = v if not state[k] else state[k]
                    elif isinstance(v, str):
                        state[k] = v if not state[k] else state[k]
                    elif isinstance(v, bool):
                        state[k] = v if state[k] == False else state[k]

    # Start the conversation
    first_msg = CONVERSATION_STATE["messages"][-1]
    print(f"\n🤖 AI: {first_msg['content']}")

    # Continue conversation until exit conditions are met
    while True:

        # Prompt user for input
        user_input = input("🗣️  User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n🤖 AI: Goodbye!")
            break

        # Add user input to messages
        CONVERSATION_STATE["messages"].append({"role": "user", "content": user_input})

        # With updates messages, stream updates
        stream_graph_updates(CONVERSATION_STATE)

        # If all required fields have been collected, end conversation
        check = [
            "customer_name",
            "product",
            "delivery_address",
            "email_address",
            "phone_number",
            "payment_method",
            "delivery_date"
        ]
        if all(CONVERSATION_STATE[key] for key in check):
            CONVERSATION_STATE["is_finished"] = True
            print("\n🤖 AI: Thanks for your time, goodbye!")
            with open("data/log.json", "w") as f:
                f.write(json.dumps(CONVERSATION_STATE, indent=2, sort_keys=True, default=str))
            break

        # If conversation is finished, eend conversation
        if CONVERSATION_STATE["is_finished"]:
            print("\n🤖 AI: Thanks for your time! Bye.")
            with open("data/log.json", "w") as f:
                f.write(json.dumps(CONVERSATION_STATE, indent=2, sort_keys=True, default=str))
            break

if __name__ == "__main__":
    main()
```

This completes the setup of our AI agent. You can now run the `main.py` file to start the application and interact with the agent.

## Happy coding! 🚀

```bash
🐢 Arun Godwin Patel @ Code Creations
```
