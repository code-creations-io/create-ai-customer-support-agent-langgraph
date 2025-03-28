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
