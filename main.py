import json

from ai.graph import State, AiAgent, CONVERSATION_STATE

def main():

    ai_agent = AiAgent()
    graph = ai_agent.build()

    # Streaming function to stream the graph updates 
    def stream_graph_updates(state: State):
        """Runs or streams the graph with the current state."""
        for event in graph.stream(state):
            for value in event.values():
                
                # Update conversation state
                for k, v in value.items():

                    # These are new messages from the tool or LLM
                    if k == "messages":
                        for msg in v:
                            # Append them to the main conversation so the full history is in CONVERSATION_STATE["messages"]
                            CONVERSATION_STATE["messages"].append(msg)
                            # Print them to the console
                            # We'll do a naive approach. If msg is a ToolMessage or LLM, show it.
                            if hasattr(msg, "content"):
                                print(f"🤖 AI: {msg.content}")
                    
                    # List
                    elif isinstance(v, list):
                        CONVERSATION_STATE[k] = v if not CONVERSATION_STATE[k] else CONVERSATION_STATE[k]

                    # String
                    elif isinstance(v, str):
                        CONVERSATION_STATE[k] = v if not CONVERSATION_STATE[k] else CONVERSATION_STATE[k]
                    
                    # Boolean
                    elif isinstance(v, bool):
                        CONVERSATION_STATE[k] = v if not CONVERSATION_STATE[k] else CONVERSATION_STATE[k]

    # Print the agent's first message
    first_msg = CONVERSATION_STATE["messages"][-1]
    print(f"🤖 AI: {first_msg['content']}")

    # Start the conversation loop
    while True:
        
        user_input = input("🗣️  User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("🤖 AI: Goodbye!")
            print(json.dumps(CONVERSATION_STATE, indent=2, sort_keys=True, default=str))
            break

        # Store user message
        CONVERSATION_STATE["messages"].append({"role": "user", "content": user_input})

        # 1) Run/stream the graph
        stream_graph_updates(CONVERSATION_STATE)

        # 2) If forcibly ended
        if CONVERSATION_STATE["is_finished"]:
            print("🤖 AI: Thanks for your time! Bye.")
            print(json.dumps(CONVERSATION_STATE, indent=2, sort_keys=True, default=str))
            break

        # 3) End if user has confirmed all checklist items
        check = ["customer_name", "product", "delivery_address", "email_address", "phone_number", "payment_method", "delivery_date"]
        if all(CONVERSATION_STATE[key] != "" for key in check):
            CONVERSATION_STATE["is_finished"] = True
            print("🤖 AI: Thanks for your time, goodbye!")
            
            # Save the conversation state to a JSON file
            with open("data/log.json", "w") as f:
                f.write(json.dumps(CONVERSATION_STATE, indent=2, sort_keys=True, default=str))
            break

if __name__ == "__main__":
    main()
