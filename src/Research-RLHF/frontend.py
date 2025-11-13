import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from research_rlhf_agent_groq import get_groq_agent
import streamlit as st
from datetime import datetime
import json

st.set_page_config(
    page_title="Agent",
    page_icon="💬",
    layout="wide"
)

checkpointer = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    st.session_state.agent = get_groq_agent(checkpointer)

config = {
    'configurable': {
        'thread_id': st.session_state.thread_id
    }
}

st.title("💬 Chat Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "insights" not in st.session_state:
    st.session_state.insights = ""

if "interrupted" not in st.session_state:
    st.session_state.interrupted = False

if "interrupt_response" not in st.session_state:
    st.session_state.interrupt_response = None

if "interrupt_value" not in st.session_state:
    st.session_state.interrupt_value = None


def get_last_message_with_content(messages_list):
    """Find the last message with actual content"""
    for msg in reversed(messages_list):
        if hasattr(msg, 'content') and msg.content and msg.content.strip():
            return msg.content
    return None


def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


display_messages()

# Handle HITL input
if st.session_state.interrupted and st.session_state.interrupt_response:
    # Display the agent's response that caused interrupt
    with st.chat_message("assistant"):
        st.markdown(st.session_state.interrupt_response)

    st.info("⏸️ Agent paused - provide feedback to continue")
    st.write("**Current Analysis:**")
    if st.session_state.interrupt_value:
        if isinstance(st.session_state.interrupt_value, dict):
            st.json(st.session_state.interrupt_value)
        else:
            st.write(st.session_state.interrupt_value[0].value)

    hitl_input = st.chat_input("Your feedback (or type 'Approved' to continue)...")

    if hitl_input:
        # Add human feedback to messages
        st.session_state.messages.append({
            "role": "user",
            "content": hitl_input,
            "timestamp": datetime.now()
        })

        with st.chat_message("user"):
            st.markdown(hitl_input)

        with st.chat_message("assistant"):
            with st.spinner("Agent processing feedback..."):
                try:
                    # Resume with same agent instance
                    response = st.session_state.agent.invoke(
                        Command(resume=hitl_input),
                        config=config
                    )

                    st.session_state.interrupted = False
                    st.session_state.interrupt_response = None
                    st.session_state.interrupt_value = None

                    # Check if still interrupted
                    if response.get("__interrupt__"):
                        # Still interrupted after resume
                        final_response = get_last_message_with_content(response['messages'])
                        if final_response:
                            st.session_state.interrupt_response = final_response
                            st.session_state.interrupted = True
                            st.session_state.interrupt_value = response.get("__interrupt__")
                        st.rerun()
                    else:
                        # Not interrupted anymore - completed
                        final_response = get_last_message_with_content(response['messages'])
                        if final_response:
                            st.markdown(final_response)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": final_response,
                                "timestamp": datetime.now()
                            })
                        st.markdown(f"\nInsights gained: {response['insights']}")
                        st.success("✅ Research completed!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback

                    st.error(traceback.format_exc())

# User input
elif prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now()
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            init_state = {
                'messages': HumanMessage(content=prompt),
                'user_input': prompt,
                'insights': st.session_state.insights,
                'first_run': True,
                'approved': False,
                'counter': 0,
                'max_loop': 2,
                'changes': [],
                'steps': '',
                'reasoning': '',
                'first_response': '',
                'current_response': '',
                'final_response': '',
                'jump_to': '',
                'content': []
            }

            try:
                response = st.session_state.agent.invoke(init_state, config=config)

                # Check if interrupted
                if response.get("__interrupt__"):
                    interrupt_msg = get_last_message_with_content(response['messages'])
                    interrupt_val = response.get("__interrupt__")

                    if interrupt_msg:
                        print(f'\n✅ Interrupt Found: {interrupt_msg}')
                        print(f'Interrupt Value: {interrupt_val}')

                        st.session_state.interrupt_response = interrupt_msg
                        st.session_state.interrupt_value = interrupt_val
                        st.session_state.interrupted = True
                        st.markdown(interrupt_msg)
                        st.warning("⏸️ Waiting for your input to continue...")
                    else:
                        st.warning("⚠️ Interrupt triggered but no message content found")

                    st.rerun()
                else:
                    # Not interrupted - completed normally
                    final_response = get_last_message_with_content(response['messages'])
                    if final_response:
                        st.markdown(final_response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_response,
                            "timestamp": datetime.now()
                        })
                    st.markdown(f"\nInsights gained: {response['insights']}")
                    st.success("✅ Research completed!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback

                st.error(traceback.format_exc())