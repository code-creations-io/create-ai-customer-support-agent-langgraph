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
MODEL = "claude-3-5-haiku-20241022"
ROLE = (
    "You are a customer support agent for a Domestic Robotics company called HomeBots. "
    "You are following up with prospective customers that have inquired for more information online. "
    "Your goal is to answer their queries and guide them through the process of purchasing a HomeBot. "
)
CONTEXT = (
    "Here is a mandatory checklist of requirements from the customer in order for them to purchase a HomeBot: "
    "\n- Customer's name"
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
    "\n2. Check the case files once (unless new info arises) to see if the customer has submitted all there are any oustanding items needed from the checklist."
    "\n3. If any checklist items are missing, one by one, ask the customer to provide each of them and save their answer. If all checklist items are saved, inform the customer that they are ready to purchase a HomeBot."
    "\n4. End the conversation."
)
RULES = (
    "- IMPORTANT: Once you have called the check_required_checklist_items tool once, do not call it again unless the user explicitly asks you to re-check. If state[\"checked_checklist\"] is True, do not call check_required_checklist_items. Instead, continue the conversation without repeating the checklist requirements."
    "\n- Use the customer's name if you have it. If the customer's name is already known in the conversation state, do not ask for it again."
    "\n- Always be polite and professional."
    "\n- Be clear and concise in your communication."
    "\n- Only focus on 1 question at a time, do not overwhelm the customer with multiple questions at once."
    "\n- Your goal is to collect all the items in the checklist. It may occur that the customer has questions or needs assistance, be prepared to address those too.The customer may provide answers to the checklist items in an unexpected order, this is fine, just make sure you have all the required information."
    "\n- If the customer has all the required information, inform them that they are ready to purchase a HomeBot."
)

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

# Tools
@tool
def check_required_checklist_items(
    state: State,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """
    This tool checks the customer inquiry to see which checklist items are currently available.
    If we've already run this tool before we do a short-circuit response to avoid repeating the same logic.
    """
    if state["checked_checklist"]:
        state_update = {
            "messages": [
                ToolMessage(
                    "We've already told you which checlist items are missing. Let's not repeat ourselves.",
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
            "checked_checklist": True,
        }
        return Command(update=state_update)

    inquiry = {}
    with open("data/customer_inquiry.json", "r") as f:
        inquiry = json.load(f)

    requirements = [
        "customer_name",
        "product",
        "delivery_address",
        "email_address",
        "phone_number",
        "payment_method",
        "delivery_date",
    ]
    missing_requirements = [item for item in requirements if not inquiry.get(item)]

    if missing_requirements:
        response = f"We need the following checklist items from you before you can purchase a HomeBot: {', '.join(missing_requirements)}"
    else:
        response = "All required checklist items have been provided."

    state_update = {
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
        "customer_name": inquiry.get("customer_name", ""),
        "product": inquiry.get("product", ""),
        "delivery_address": inquiry.get("delivery_address", ""),
        "email_address": inquiry.get("email_address", ""),
        "phone_number": inquiry.get("phone_number", ""),
        "payment_method": inquiry.get("payment_method", ""),
        "delivery_date": inquiry.get("delivery_date", ""),
        "is_finished": False,
        "checked_checklist": True,
    }
    return Command(update=state_update)

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
        "delivery_date": iso_datetime,
        "messages": [
            ToolMessage(
                content=f"Great, I'll set the delivery date as {iso_datetime}.",
                tool_call_id=tool_call_id,
            )
        ],
    }
    return Command(update=state_update)


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
