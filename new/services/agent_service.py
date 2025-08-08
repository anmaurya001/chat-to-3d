"""Agent service for LLM integration."""

import logging
import requests
import json
from datetime import datetime
from enum import Enum
from griptape.structures import Agent
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.memory.structure import ConversationMemory
from griptape.rules import Rule
import config

# Set up logging
logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
AGENT_MODEL = config.AGENT_MODEL
AGENT_BASE_URL = config.AGENT_BASE_URL


class RuleType(Enum):
    """Enum for different rule types."""
    PLANNING = "planning"
    PROMPT_GENERATION = "prompt_generation"


class LLMAgent:
    """LLM agent using Griptape framework."""
    
    def __init__(self):
        self.memory = ConversationMemory()
        self.agent = self._initialize_agent()
        self.is_generating_prompts = False
        self.health_check_url = f"{AGENT_BASE_URL}/health/ready"
    
    def _get_planning_rules(self):
        """Get rules for scene planning phase."""
        return [
            Rule("You are a helpful scene planning assistant for 3D content creation."),
            Rule("Your primary task is to suggest objects for the 3D scene based on user requests."),
            Rule("Always suggest objects in singular form (e.g., 'Palm Tree' instead of 'Palm Trees')."),
            Rule("Always format object names with proper capitalization and spaces (e.g., 'Coffee Table' not 'coffee_table')."),
            Rule(f"When suggesting objects for a general scene request: Suggest exactly {config.NUM_OF_OBJECTS} objects that would be appropriate for this scene."),
            Rule(f"Always format your object suggestions in this exact format: Suggested objects: 1. object_name 2. object_name ... {config.NUM_OF_OBJECTS}. object_name"),
            Rule("Focus only on suggesting appropriate objects for the scene based on the user's requests."),
        ]
    
    def _get_prompt_generation_rules(self):
        """Get rules for 2D prompt generation phase."""
        return [
            Rule("You are now in 2D prompt generation mode. Your task is to create detailed prompts for each object in the scene."),
            Rule("The descriptions should be highly detailed and visually rich, suitable for a text-to-image generation model."),
            Rule("The prompt must specify a plain or empty background (e.g., 'on a white background', 'isolated', 'no background')."),
            Rule("Focus ONLY on the physical and visual characteristics of each object itself."),
            Rule(f"Keep each object's prompt to exactly {config.TWO_D_PROMPT_LENGTH} words or less for optimal generation quality."),
            Rule("Generate a separate prompt for each object in the scene."),
            Rule("Format each object's prompt with 'Object:' and 'Prompt:' labels."),
            Rule("Do not add any explanatory notes or comments after the prompt."),
            Rule("Do not use asterisks or any special formatting characters."),
            Rule("Output only the Object and Prompt labels with clean text - no additional formatting or notes."),
            Rule(
                "The prompt text should describe the object's visual characteristics in detail."
                "\nExample:"
                "\nObject: Beach Chair"
                "\nPrompt: A comfortable beach chair with ergonomic design and colorful fabric, on a white background"
                "\nExample:"
                "\nObject: Beach Umbrella"
                "\nPrompt: A vibrant beach umbrella with colorful stripes and sturdy metal frame, on a white background"
            ),
        ]
    
    def _initialize_agent(self):
        """Initialize the agent with initial planning rules."""
        try:
            logger.info(f"Initializing agent with base URL: {AGENT_BASE_URL}")
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
    
    def run(self, prompt, rule_type=RuleType.PLANNING):
        """Run a prompt through the LLM with specified rule type."""
        try:
            # Set the appropriate rules based on rule type
            if rule_type == RuleType.PROMPT_GENERATION:
                self.agent.rules = self._get_prompt_generation_rules()
                self.is_generating_prompts = True
            else:
                self.agent.rules = self._get_planning_rules()
                self.is_generating_prompts = False
            
            response = self.agent.run(prompt)
            return response
        except Exception as e:
            logger.error(f"Error running prompt: {e}")
            raise
    
    def check_agent_health(self):
        """Check if the LLM agent is up and running."""
        try:
            logger.info(f"Checking agent health at: {self.health_check_url}")
            response = requests.get(self.health_check_url, timeout=2)
            logger.info(f"Agent health check response: {response.status_code}")
            return response.status_code == 200
        except requests.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory = ConversationMemory()
        self.agent.memory = self.memory
        # Reinitialize the agent to get a fresh context
        self.agent = self._initialize_agent()


class AgentService:
    """Service class for managing LLM agent interactions."""
    
    def __init__(self):
        """Initialize the agent service."""
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the planning agent."""
        try:
            self.agent = LLMAgent()
            print("✅ LLM agent initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LLM agent: {e}")
            raise
    
    def is_healthy(self):
        """Check if the agent is healthy and ready."""
        return self.agent.check_agent_health()
    
    def chat(self, message, current_objects=None):
        """Send a chat message to the agent."""
        try:
            response = self.agent.run(message)
            return response.output.value
        except Exception as e:
            return f"Error communicating with agent: {str(e)}"
    
    def generate_objects_for_scene(self, scene_description):
        """Generate 20 singular objects for a given scene description."""
        try:
            # Create a specific prompt for generating 20 objects
            prompt = f"""Based on this scene description: "{scene_description}"

Please suggest exactly {config.NUM_OF_OBJECTS} objects that would be appropriate for this scene. 

Requirements:
- Suggest exactly {config.NUM_OF_OBJECTS} objects (no more, no less)
- Use singular form for all objects (e.g., 'Chair' not 'Chairs')
- Use proper capitalization and spaces (e.g., 'Coffee Table' not 'coffee_table')
- Focus on objects that would realistically be found in this type of scene
- Ensure variety and complementarity between objects

Format your response as:
Suggested objects:
1. object_name
2. object_name
3. object_name
...
{config.NUM_OF_OBJECTS}. object_name

Scene arrangement: [Brief description of how these objects could be arranged together]"""

            response = self.agent.run(prompt, RuleType.PLANNING)
            response_text = response.output.value
            
            # Parse the response to extract object names
            objects = self._parse_objects_from_response(response_text)
            return objects
            
        except Exception as e:
            print(f"Error generating objects: {e}")
            return []
    
    def _parse_objects_from_response(self, response_text):
        """Parse object names from the agent's response."""
        objects = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', 
                                        '11.', '12.', '13.', '14.', '15.', '16.', '17.', '18.', '19.', '20.'))):
                # Extract object name after the number
                object_name = line.split('.', 1)[1].strip()
                if object_name:
                    objects.append(object_name)
        
        return objects
    
    
    def generate_objects_and_prompts(self, description):
        """Generate objects and 2D prompts for the scene objects."""
        try:
            # Get objects for the scene
            objects = self.generate_objects_for_scene(description)
            if not objects:
                return False, None, "No objects generated"
            
            prompts = {}
            for obj in objects:
                prompt = f"Generate visual prompt suitable for 2D image generation for: {obj}"
                response = self.agent.run(prompt, RuleType.PROMPT_GENERATION)
                response_text = response.output.value
                
                # Extract the prompt from the response
                if "Object:" in response_text and "Prompt:" in response_text:
                    prompt_text = response_text.split("Prompt:")[-1].strip()
                    prompts[obj] = prompt_text
                else:
                    prompts[obj] = f"{obj}, detailed 2D image on white background"
            
            return True, prompts, "2D prompts generated successfully"
        except Exception as e:
            return False, None, f"Error generating prompts: {str(e)}"
    
    def clear_memory(self):
        """Clear the agent's conversation memory."""
        self.agent.clear_memory()
        return True 