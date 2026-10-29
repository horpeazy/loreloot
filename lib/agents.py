from pathlib import Path
from typing import TypedDict, List, Optional, Union, TypeVar
from pydantic import BaseModel
from tavily import TavilyClient
import json

from lib.state_machine import StateMachine, Step, EntryPoint, Termination, Run
from lib.llm import LLM
from lib.messages import AIMessage, UserMessage, SystemMessage, ToolMessage, BaseMessage
from lib.tooling import Tool, ToolCall
from lib.memory import ShortTermMemory, LongTermMemory, MemoryFragment
from lib.vector_db import VectorStoreManager,QueryResult

class EvaluationResponse(BaseModel):
    useful: bool
    description: str

class EvaluationData(TypedDict):
    useful: bool
    description: str

class WebResultData(TypedDict):
    title: str
    url: str
    content: str

# Define the state schema
class AgentState(TypedDict):
    user_query: str  # The current user query being processed
    instructions: str  # System instructions for the agent
    messages: List[dict]  # List of conversation messages
    current_tool_calls: Optional[List[ToolCall]]  # Current pending tool calls
    total_tokens: int  # Track the cumulative total

class ResearchAgentState(TypedDict):
    user_query: str  # The current user query being processed
    instructions: str  # System instructions for the agent
    messages: List[BaseMessage]  # List of conversation messages
    current_tool_calls: Optional[List[ToolCall]]  # Current pending tool calls
    session_id: str
    documents: List[str]
    evaluation: Optional[EvaluationData]
    web_results: List[WebResultData]
    total_tokens: int  # Track the cumulative total
    
class Agent:
    def __init__(self, 
                 model_name: str,
                 instructions: str, 
                 tools: List[Tool] = None,
                 temperature: float = 0.7):
        """
        Initialize an Agent
        
        Args:
            model_name: Name/identifier of the LLM model to use
            instructions: System instructions for the agent
            tools: Optional list of tools available to the agent
            temperature: Temperature parameter for LLM (default: 0.7)
        """
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize memory and state machine
        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """Step logic: Prepare messages for LLM consumption"""
        messages = state.get("messages", [])
        
        # If no messages exist, start with system message
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
            
        # Add the new user message
        messages.append(UserMessage(content=state["user_query"]))
        
        return {
            "messages": messages,
            "session_id": state["session_id"]
        }

    def _llm_step(self, state: AgentState) -> AgentState:
        """Step logic: Process the current state through the LLM"""
        # Initialize LLM
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            tools=self.tools,
        )

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        # Create AI message with content and tool calls
        ai_message = AIMessage(
            content=response.content, 
            tool_calls=tool_calls,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        """Step logic: Execute any pending tool calls"""
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []
        
        for call in tool_calls:
            # Access tool call data correctly
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id
            # Find the matching tool
            tool = next((t for t in self.tools if t.name == function_name), None)
            if tool:
                result = str(tool(**function_args))
                tool_message = ToolMessage(
                    content=json.dumps(result), 
                    tool_call_id=tool_call_id, 
                    name=function_name, 
                )
                tool_messages.append(tool_message)
        
        # Clear tool calls and add results to messages
        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"]
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Create the internal state machine for the agent"""
        machine = StateMachine[AgentState](AgentState)
        
        # Create steps
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()
        
        machine.add_steps([entry, message_prep, llm_processor, tool_executor, termination])
        
        # Add transitions
        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)
        
        # Transition based on whether there are tool calls
        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            """Transition logic: Check if there are tool calls"""
            if state.get("current_tool_calls"):
                return tool_executor
            return termination
        
        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(tool_executor, llm_processor)  # Go back to llm after tool execution
        
        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query
        
        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses "default" if None)
            
        Returns:
            The final run object after processing
        """
        session_id = session_id or "default"

        # Create session if it doesn't exist
        self.memory.create_session(session_id)

        # Get previous messages from last run if available
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
        }

        run_object = self.workflow.run(initial_state)
        
        # Store the complete run object in memory
        self.memory.add(run_object, session_id)
        
        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """Get all Run objects for a session
        
        Args:
            session_id: Optional session ID (uses "default" if None)
            
        Returns:
            List of Run objects in the session
        """
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """Reset memory for a specific session
        
        Args:
            session_id: Optional session to reset (uses "default" if None)
        """
        self.memory.reset(session_id)

class ResearchAgent:
    def __init__(self,
                 model_name: str,
                 instructions: str,
                 api_key: str,
                 api_base: str,
                 tavily_api_key: str,
                 temperature: float = 0.7,
                 storage_path: str = ".",
                 memory_owner: str = "udaplay"):
        self.memory_owner = memory_owner
        self.instructions = instructions
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.tavily_api_key = tavily_api_key

        self.ltm_db = VectorStoreManager(str(Path(storage_path) / "ltm"), self.api_key, self.api_base)
        self.games_db = VectorStoreManager(str(Path(storage_path) / "chromadb"), self.api_key, self.api_base)
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory(self.ltm_db)
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """Prepare the conversation messages for the current question."""
        messages = state.get("messages", [])
        if not messages:
            messages = [SystemMessage(content=state["instructions"])]
        messages.append(UserMessage(content=state["user_query"]))
        return {
            "messages": messages,
            "session_id": state["session_id"],
        }

    def _llm_step(self, state: ResearchAgentState) -> ResearchAgentState:
        """Include retrieved evidence and generate the final answer."""
        llm = LLM(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
        )
        documents = "\n".join(state["documents"])
        retrieval_aug_text = (
            f"Local Data: {documents}\n\n"
            f"Web Results: {json.dumps(state['web_results'])}\n\n"
            f"Evaluation: {json.dumps(state['evaluation'])}"
        )
        llm_messages = [
            *state["messages"],
            UserMessage(content=retrieval_aug_text),
        ]
        response = llm.invoke(llm_messages)
        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens
        return {
            "messages": state["messages"] + [AIMessage(content=response.content)],
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _retrieve_game(self, state: ResearchAgentState) -> ResearchAgentState:
        """Search game documents and saved research for the question."""
        query = state["user_query"]
        games_collection = self.games_db.get_or_create_store("udaplay")
        result = games_collection.query(query_texts=[query], n_results=3)
        game_docs = result["documents"][0]
        memory_result = self.long_term_memory.search(
            query, self.memory_owner, namespace="game_research"
        )
        memory_docs = [
            f"Title: {fragment.metadata.get('title', '')} - "
            f"Url: {fragment.metadata.get('url', '')} - Content: {fragment.content}"
            for fragment in memory_result.fragments
        ]
        return {
            "documents": game_docs + memory_docs,
            "session_id": state["session_id"],
        }

    def _evaluate_doc_retrieval(self, state: ResearchAgentState) -> ResearchAgentState:
        """Evaluate whether the internal documents answer the question."""
        query = state["user_query"]
        documents = "\n".join(state["documents"])
        llm = LLM(api_key=self.api_key)
        prompt = (
            "Your task is to evaluate whether the documents are sufficient to answer the query. "
            "Give a detailed explanation so someone can decide whether to accept the answer.\n\n"
            f"Query: {query}\nDocuments: {documents}"
        )
        response = llm.invoke(prompt, EvaluationResponse)
        evaluation = EvaluationResponse.model_validate_json(response.content)
        return {
            "evaluation": evaluation.model_dump(),
            "session_id": state["session_id"],
        }

    def _evaluate_web_retrieval(self, state: ResearchAgentState) -> ResearchAgentState:
        """Evaluate the combined evidence and save useful web research."""
        query = state["user_query"]
        documents = [r["content"] for r in state["web_results"]] + state["documents"]
        evidence = "\n".join(documents)
        llm = LLM(api_key=self.api_key)
        prompt = (
            "Your task is to evaluate whether the documents are sufficient to answer the query. "
            "Give a detailed explanation so someone can decide whether to accept the answer.\n\n"
            f"Query: {query}\nDocuments: {evidence}"
        )
        response = llm.invoke(prompt, EvaluationResponse)
        evaluation = EvaluationResponse.model_validate_json(response.content).model_dump()
        if evaluation["useful"]:
            for result in state["web_results"]:
                fragment = MemoryFragment(
                    content=result["content"],
                    owner=self.memory_owner,
                    namespace="game_research",
                )
                self.long_term_memory.register(
                    fragment,
                    metadata={"title": result["title"], "url": result["url"]},
                )
        return {
            "evaluation": evaluation,
            "session_id": state["session_id"],
        }

    def _game_web_search(self, state: ResearchAgentState) -> ResearchAgentState:
        """Search the web when local evidence is insufficient."""
        response = self.tavily_client.search(
            query=state["user_query"],
            max_results=3,
        )
        web_results = [
            {"title": result["title"], "url": result["url"], "content": result["content"]}
            for result in response["results"]
        ]
        return {
            "web_results": web_results,
            "session_id": state["session_id"],
        }

    def _create_state_machine(self) -> StateMachine[ResearchAgentState]:
        machine = StateMachine[ResearchAgentState](ResearchAgentState)
        entry = EntryPoint[ResearchAgentState]()
        message_prep = Step[ResearchAgentState]("message_prep", self._prepare_messages_step)
        retrieve_game = Step[ResearchAgentState]("retrieve_game", self._retrieve_game)
        evaluate_doc_retrieval = Step[ResearchAgentState]("_evaluate_doc_retrieval", self._evaluate_doc_retrieval)
        game_web_search = Step[ResearchAgentState]("game_web_search", self._game_web_search)
        evaluate_web_retrieval = Step[ResearchAgentState]("evaluate_web_retrieval", self._evaluate_web_retrieval)
        llm_processor = Step[ResearchAgentState]("llm_processor", self._llm_step)
        termination = Termination[ResearchAgentState]()
        machine.add_steps([
            entry, message_prep, retrieve_game, evaluate_doc_retrieval,
            game_web_search, evaluate_web_retrieval, llm_processor, termination,
        ])

        def check_doc_evaluation_result(state: ResearchAgentState):
            if state["evaluation"]["useful"]:
                return llm_processor
            return game_web_search

        machine.connect(entry, message_prep)
        machine.connect(message_prep, retrieve_game)
        machine.connect(retrieve_game, evaluate_doc_retrieval)
        machine.connect(evaluate_doc_retrieval, [llm_processor, game_web_search], check_doc_evaluation_result)
        machine.connect(game_web_search, evaluate_web_retrieval)
        machine.connect(evaluate_web_retrieval, llm_processor)
        machine.connect(llm_processor, termination)
        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        session_id = session_id or "default"
        self.short_term_memory.create_session(session_id)
        previous_messages = []
        last_run = self.short_term_memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]
        initial_state: ResearchAgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
            "documents": [],
            "evaluation": None,
            "web_results": [],
            "total_tokens": 0,
        }
        run_object = self.workflow.run(initial_state)
        self.short_term_memory.add(run_object, session_id)
        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        return self.short_term_memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        self.short_term_memory.reset(session_id)
