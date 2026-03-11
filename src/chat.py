import json
import re
from huggingface_hub import InferenceClient

from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from src.search_criteria import SearchCriteria, FacilityOption, ready_to_search
from src.facility_search import search as facility_search

# Keys we collect from the user for facility search (must match SearchCriteria)
CRITERIA_KEYS = [
    "location_city", "location_state", "location_zip",
    "treatment_type", "substances", "payment_options",
    "special_populations", "therapies", "languages",
]


class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """
    memory = []

    task = """
            Help individuals find appropriate substance use and mental health treatment facilities. 
            You should help someone describe their situation and find facilities that match their needs.
            Do not give the user the option to search for facilities until you have all the information you need.
            Make sure to first ask claryfying questions to help you understand the user's situation, including location, treatment type, payment options, special populations, therapies, and languages.
            Do not ask too many questions at once.
            Ask one question at a time.

            Assume the user is located in Boston, Massachusetts, United States.
            Assume that the user is going to give you responses.
            Then, you should help the user find facilities that match their needs.
            When you are given facility options to show (by the system), explain each option clearly: name, key features, and how it might fit the user. 
            Default location if not given: Boston, Massachusetts.

            If the user did not ask for a facility at first but the conversation suggests they could benefit (substance use, mental health, needing treatment), 
            gently suggest finding a facility and ask clarifying questions that would help find a facility that matches the user's needs one at a time. 
            Once you have at least location AND treatment type, tell the user you have enough to look up some options and that you can share a few matches next.
            IMPORTANT: Never mix “I have enough info” with asking another question in the same message. If you ask a question, you do NOT yet have enough.

            SAMHSA’s directory contains thousands of treatment facilities nationwide, each with many attributes: treatment
            type (inpatient, outpatient, residential, telehealth), substances addressed, payment options
            (Medicaid, sliding scale, free, private insurance), special populations served (veterans, LGBTQ+,
            adolescents, pregnant women), therapies offered (CBT, MAT, 12-step), and languages spoken. 
        """

    # Controller state (prevents repeated auto-search)
    has_searched = False

    def add_to_memory(self, user_input, response):
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": response})

    def get_memory(self):
        return self.memory

    def clear_memory(self):
        self.memory = []
        self.has_searched = False

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        self.clear_memory()
        self.memory.append({"role": "system", "content": self.task})
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL # define MY_MODEL in config.py if you create a new model in the HuggingFace Hub
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        
    def format_prompt(self, user_input):
        """
        TODO: Implement this method to format the user's input into a proper prompt.
        
        This method should:
        1. Add any necessary system context or instructions
        2. Format the user's input appropriately
        3. Add any special tokens or formatting the model expects

        Args:
            user_input (str): The user's question

        Returns:
            str: A formatted prompt ready for the model
        
        Example prompt format:
            "You are a helpful assistant that specializes in...
             User: {user_input}
             Assistant:"
        """
        return user_input
        
    def get_response(self, user_input):
        """
        TODO: Implement this method to generate responses to user questions.
        
        This method should:
        1. Use format_prompt() to prepare the input
        2. Generate a response using the model
        3. Clean up and return the response

        Args:
            user_input (str): The user's question

        Returns:
            str: The chatbot's response

        Implementation tips:
        - Use self.format_prompt() to format the user's input
        - Use self.client to generate responses
        """
        # TODO: use tools to go to the course catalog
        try:
            formatted_input = self.format_prompt(user_input)
            
            messages= self.memory + [
                {"role": "user", "content": formatted_input},
            ]
            output = self.client.chat.completions.create(
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=messages,
                        max_tokens=512,
                    )
            response = (output.choices[0].message.content or "").strip()
            self.add_to_memory(formatted_input, response)

            # Deterministic auto-search:
            # - never auto-search if assistant is still asking a question
            # - only auto-search once per conversation unless user explicitly triggers again
            assistant_is_asking = "?" in response
            if (not self.has_searched) and (not assistant_is_asking):
                criteria = self.get_search_criteria()
                if ready_to_search(criteria):
                    options = facility_search(criteria)
                    explanation = self.explain_facility_options(options)
                    combined = response + "\n\n" + explanation
                    self.memory[-1]["content"] = combined
                    self.has_searched = True
                    return combined

            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def get_search_criteria(self) -> SearchCriteria:
        """
        Extract key information from the conversation so the facility search module can run a search.
        The external search module (to be implemented) should call this and use the returned dict.

        When to run:
          - When the user asks to search (e.g. "search for facilities", "find options", or a
            "Search" button click).
          - After that, use ready_to_search(criteria) to check if you have at least location +
            treatment_type before calling the search module; if not, ask for the missing info.

        Returns:
            SearchCriteria: Dict with keys such as location_city, location_state, treatment_type,
                payment_options, special_populations, therapies, languages, substances.
                Only keys that could be inferred from the conversation are present.
        """
        if not self.memory or len(self.memory) < 2:
            return {}
        extract_instruction = """From the conversation below, extract the user's preferences for finding a treatment facility. Return a single JSON object only, no other text. Use these exact keys when present: location_city, location_state, location_zip, treatment_type (one of: inpatient, outpatient, residential, telehealth), payment_options (list), special_populations (list), therapies (list), languages (list), substances (list). Use empty list [] or omit the key if unknown. Examples: "Boston" -> location_city, "MA" or "Massachusetts" -> location_state."""
        conv_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.memory if m["role"] in ("user", "assistant")
        )
        messages = [
            {"role": "user", "content": extract_instruction + "\n\nConversation:\n" + conv_text},
        ]
        try:
            output = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=messages,
                max_tokens=512,
            )
            raw = output.choices[0].message.content.strip()
            # Try to pull out a JSON object if the model wrapped it in markdown or text
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                raw = json_match.group(0)
            data = json.loads(raw)
            return {k: v for k, v in data.items() if k in CRITERIA_KEYS and v is not None}
        except (json.JSONDecodeError, Exception):
            return {}

    def explain_facility_options(self, options: list[FacilityOption]) -> str:
        """
        Explain the facility options returned by the search module in a user-friendly way.
        Call this after the external system returns a list of facilities.

        Args:
            options: List of facility dicts with keys such as name, address, phone,
                treatment_types, payment_options, special_populations, therapies, languages.

        Returns:
            str: A conversational explanation of the options for the user.
        """
        if not options:
            return "No facilities matched your criteria. We can try adjusting location or preferences—tell me what you'd like to change."
        options_text = json.dumps(options, indent=2)
        prompt = f"""Based on the user's conversation and these facility results, explain each option clearly and briefly: name, location if present, key features (treatment type, payment, special populations, therapies, languages), and how it might fit their situation. Be warm and helpful.\n\nFacility options:\n{options_text}"""
        messages = self.memory + [{"role": "user", "content": prompt}]
        try:
            output = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                messages=messages,
                max_tokens=1024,
            )
            return output.choices[0].message.content
        except Exception as e:
            return f"I found {len(options)} option(s), but couldn't generate a summary right now: {e}. Here are the names: " + ", ".join(
                str(o.get("name", "Unknown")) for o in options
            )
