"""
Gradio Web Interface for Boston School Chatbot

This script creates a web interface for your chatbot using Gradio.
You only need to implement the chat function.

Key Features:
- Creates a web UI for your chatbot
- Handles conversation history
- Provides example questions
- Can be deployed to Hugging Face Spaces

Example Usage:
    # Run locally:
    python app.py
    
    # Access in browser:
    # http://localhost:7860
"""

import gradio as gr
from src.chat import Chatbot
from src.facility_search import search as facility_search
from src.search_criteria import ready_to_search

# Run get_search_criteria() when the user asks to search (trigger phrase or "Search" button).
# After getting criteria, check ready_to_search(criteria) before calling the search module.
SEARCH_TRIGGERS = ("search for facilities", "find facilities", "look for facilities", "ready to search", "run the search", "search now")


def create_chatbot():
    """
    Creates and configures the chatbot interface.
    """
    chatbot = Chatbot()
    
    def chat(message, history):
        """
        TODO:Generate a response for the current message in a Gradio chat interface.
        
        This function is called by Gradio's ChatInterface every time a user sends a message.
        You only need to generate and return the assistant's response - Gradio handles the
        chat display and history management automatically.

        Args:
            message (str): The current message from the user
            history (list): List of previous message pairs, where each pair is
                           [user_message, assistant_message]
                           Example:
                           [
                               ["What schools offer Spanish?", "The Hernandez School..."],
                               ["Where is it located?", "The Hernandez School is in Roxbury..."]
                           ]

        Returns:
            str: The assistant's response to the current message.


        Note:
            - Gradio automatically:
                - Displays the user's message
                - Displays your returned response
                - Updates the chat history
                - Maintains the chat interface
            - You only need to:
                - Generate an appropriate response to the current message
                - Return that response as a string
        """
        msg_lower = (message or "").strip().lower()
        if any(trigger in msg_lower for trigger in SEARCH_TRIGGERS):
            criteria = chatbot.get_search_criteria()
            if not ready_to_search(criteria):
                missing = []
                if not any((criteria.get("location_city"), criteria.get("location_state"), criteria.get("location_zip"))):
                    missing.append("location (e.g. city or state)")
                if not criteria.get("treatment_type"):
                    missing.append("treatment type (e.g. outpatient, inpatient)")
                reply = f"I'd like to search for you. I still need: {', '.join(missing)}. Share that and then say \"search for facilities\" again."
                chatbot.add_to_memory(message, reply)
                return reply
            options = facility_search(criteria)
            explanation = chatbot.explain_facility_options(options)
            chatbot.add_to_memory(message, explanation)
            chatbot.has_searched = True
            return explanation
        return chatbot.get_response(message)

    
    
    # Create Gradio interface. Customize the interface however you'd like!
    demo = gr.ChatInterface(
        chat,
        title="6.C395",
        description="I help find substance use and mental health treatment facilities. You can ask for options, or we can just talk—if it seems a facility could help, I'll suggest it and ask a few questions (location, treatment type, payment, etc.). Once I have enough info I'll search and explain options. You can also say \"search for facilities\" anytime. Free tier may show 503 when busy—try again in a few seconds.",
        examples=[
            "What options are available for someone in my situation?",
            "I've been really stressed and thinking about talking to someone.",
            "I'm in Boston and need outpatient counseling. I have Medicaid.",
            "I'm ready—search for facilities",
        ]
    )
    
    return demo

if __name__ == "__main__":
    demo = create_chatbot()
    demo.launch()
