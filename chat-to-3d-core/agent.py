import logging
from datetime import datetime
from griptape.structures import Agent
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.memory.structure import ConversationMemory
from griptape.rules import Rule
from config import AGENT_MODEL, AGENT_BASE_URL, LOG_LEVEL, LOG_FORMAT
from utils import save_prompts_to_json

# Set up logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ScenePlanningAgent:
    def __init__(self):
        self.memory = ConversationMemory()
        self.agent = self._initialize_agent()
        self.is_generating_prompts = False

    def _get_planning_rules(self):
        """Get rules for scene planning phase."""
        return [
            Rule(
                "You are a helpful scene planning assistant for 3D content creation."
            ),
            Rule(
                "Your primary task is to suggest objects for the 3D scene based on user requests."
            ),
            Rule(
                "Always format object names with proper capitalization and spaces (e.g., 'Coffee Table' not 'coffee_table' or 'coffee table')."
            ),
            Rule(
                "When suggesting objects for a general scene request:"
                "\n- Suggest between 5 to 8 objects"
                "\n- Ensure the objects complement each other"
                "\n- Consider the scene's purpose and style"
            ),
            Rule(
                "When the user requests specific objects (e.g., 'add a sofa' or 'I want a table'):"
                "\n- Focus only on the requested object type"
                "\n- Suggest 1-3 variations of the requested object"
                "\n- Keep suggestions relevant to the specific request"
                "\n- DO NOT suggest the same object type multiple times"
            ),
            Rule(
                "Before suggesting objects:"
                "\n- Check if the requested object is already in the scene"
                "\n- If it is, acknowledge it and suggest complementary objects instead"
                "\n- If it's not, suggest appropriate variations"
            ),
            Rule(
                "Always format your object suggestions in this exact format:"
                "\nSuggested objects:"
                "\n1. object_name"
                "\n2. object_name"
                "\n3. object_name"
                "\n\n\nScene arrangement: [Brief description of how these objects could be arranged together]"
            ),
            Rule(
                "After suggesting objects, always remind the user:"
                "\n'You can select which objects to keep by checking their boxes in the Object Management panel on the right. Would you like to:'"
                "\n1. Generate 3D prompts for your selected objects (click the 'Generate 3D Prompts' button)"
                "\n2. Get more suggestions for the scene"
                "\n3. Make any other changes to the scene"
            ),
            Rule(
                "If the user directly requests a specific object (e.g., 'add a sofa' or 'I want a table'), include it in the suggested objects list in the same format."
            ),
            Rule(
                "Keep object names simple, clear, and consistent. Use the same name format for the same object type."
            ),
            Rule(
                "When suggesting new objects, consider what objects are already in the scene."
            ),
            Rule(
                "If the user asks for more suggestions, provide new objects that complement the existing ones."
            ),
            Rule(
                "If the user asks to remove objects, acknowledge the request but let them use the interface checkboxes to manage objects."
            ),
            Rule(
                "DO NOT try to track or manage the object list yourself - the interface handles this."
            ),
            Rule(
                "DO NOT describe visual or physical characteristics of objects during the planning phase."
            ),
            Rule(
                "DO NOT discuss materials, textures, or surface properties during the planning phase."
            ),
            Rule(
                "Focus only on suggesting appropriate objects for the scene based on the user's requests."
            ),
            Rule(
                "When the user is satisfied with the scene, tell them they can click the 'Generate 3D Prompts' button."
            ),
            Rule(
                "If the user asks to generate 3D prompts in chat, remind them to use the 'Generate 3D Prompts' button instead."
            ),
        ]

    def _get_prompt_generation_rules(self):
        """Get rules for 3D prompt generation phase."""
        return [
            Rule(
                "You are now in 3D prompt generation mode. Your task is to create detailed prompts for each object in the scene."
            ),
            Rule(
                "For each object, focus on key visual and physical characteristics."
            ),
            Rule(
                "Include details about the object's shape, materials, and surface properties."
            ),
            Rule(
                "DO NOT include any measurements or specific dimensions in the prompts."
            ),
            Rule(
                "Incorporate the object's context and relationship to other objects in the scene."
            ),
            Rule(
                "Keep each object's prompt to exactly 40 words or less for optimal generation quality."
            ),
            Rule(
                "Use descriptive adjectives and specific material terms to enhance the 3D generation."
            ),
            Rule(
                "Generate a separate prompt for each object in the scene."
            ),
            Rule(
                "Format each object's prompt with 'Object:' and 'Prompt:' labels."
            ),
        ]

    def _initialize_agent(self):
        """Initialize the agent with initial planning rules."""
        try:
            prompt_driver = OpenAiChatPromptDriver(
                model=AGENT_MODEL,
                base_url=AGENT_BASE_URL,
                api_key="not-needed",
                user="user",
            )
        except Exception as e:
            logger.error(f"Failed to initialize prompt driver: {e}")
            raise

        agent = Agent(
            prompt_driver=prompt_driver,
            rules=self._get_planning_rules(),
        )
        agent.memory = self.memory
        return agent

    def chat(self, message, current_objects=None):
        """Handle chat messages and provide scene planning assistance."""
        try:
            # Always ensure we're in planning mode
            if self.is_generating_prompts:
                self.agent.rules = self._get_planning_rules()
                self.is_generating_prompts = False

            # Add current objects context to the message if provided
            if current_objects:
                context_message = f"Current objects in scene: {', '.join(current_objects)}\n\nUser message: {message}"
            else:
                context_message = message

            # Normal chat response - always in planning mode
            self.agent.rules = self._get_planning_rules()
            response = self.agent.run(context_message)
            response_text = response.output.value if hasattr(response, "output") else str(response)
            return response_text
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"An error occurred: {str(e)}"

    def generate_scene_name(self, description):
        """Generate a scene name from the description."""
        try:
            # response = self.agent.run(f"""Based on this scene description: "{description}"
            # Generate a short, descriptive name for this scene (2-3 words).
            # Format the response as: "Scene Name: [name]"
            # """)

            response = self.agent.run(f"""Based on this scene discussion,
            Generate a short, descriptive name for this scene (2-3 words).
            Format the response as: "Scene Name: [name]"
            """)

            response_text = (
                response.output.value if hasattr(response, "output") else str(response)
            )

            for line in response_text.split("\n"):
                if "Scene Name:" in line:
                    return line.split("Scene Name:")[-1].strip()

            return f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except Exception as e:
            logger.error(f"Error generating scene name: {e}")
            return f"scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def generate_3d_prompts(self, scene_name, initial_description):
        """Generate 3D prompts for each object in the final scene."""
        try:
            logger.info("Switching to prompt generation mode")
            # Switch to prompt generation mode
            self.agent.rules = self._get_prompt_generation_rules()
            self.is_generating_prompts = True

            generation_prompt = f"""Generate detailed 3D prompts for each object in the current scene, focusing on:
            1. Visual and physical characteristics
            2. Materials and surface properties
            3. Context within the scene
            Generate prompt for each object strictly from this list: {initial_description}
            Keep each prompt to exactly 40 words or less.
            Format each object's prompt with 'Object:' and 'Prompt:' labels.
            """

            response = self.agent.run(generation_prompt)

            response_text = (
                response.output.value if hasattr(response, "output") else str(response)
            )
            logger.info("Received response from agent")

            object_prompts = {}
            current_object = None
            current_prompt = []

            for curr_line in response_text.split("\n"):
                line = curr_line.strip()
                if not line:
                    continue

                if "Object:" in line:
                    if current_object and current_prompt:
                        prompt_text = " ".join(current_prompt)
                        word_count = len(prompt_text.split())
                        if word_count > 40:
                            prompt_text = " ".join(prompt_text.split()[:40])
                        object_prompts[current_object] = prompt_text
                    current_object = (
                        line.split("Object:")[-1].strip().strip("*").strip()
                    )
                    current_prompt = []
                elif "Prompt:" in line:
                    prompt_text = line.split("Prompt:")[-1].strip().strip("*").strip()
                    current_prompt.append(prompt_text)

            if current_object and current_prompt:
                prompt_text = " ".join(current_prompt)
                word_count = len(prompt_text.split())
                if word_count > 40:
                    prompt_text = " ".join(prompt_text.split()[:40])
                object_prompts[current_object] = prompt_text

            if not object_prompts:
                logger.warning("No prompts were extracted from the response")
                return False, None, generation_prompt

            if not save_prompts_to_json(
                scene_name, object_prompts, initial_description
            ):
                return False, None, generation_prompt

            return True, object_prompts, generation_prompt
        except Exception as e:
            logger.error(f"Error generating prompts: {e}")
            return False, None, generation_prompt
        finally:
            # Always switch back to planning mode after generation
            logger.info("Switching back to planning mode after generation")
            self.agent.rules = self._get_planning_rules()
            self.is_generating_prompts = False
            

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory = ConversationMemory()
        self.agent.memory = self.memory
        # Reinitialize the agent to get a fresh context
        self.agent = self._initialize_agent()

