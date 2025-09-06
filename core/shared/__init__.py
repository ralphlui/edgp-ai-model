"""
Core Shared Features

Shared components for agent collaboration and feature reuse.
Provides standardized communication, prompt engineering, RAG, memory management,
knowledge base management, tool management, and context management.
"""

# Communication system - always import this first
from .communication import (
    StandardAgentInput,
    StandardAgentOutput,
    ValidationError,
    Priority,
    MessageType,
    Status,
    TraceInfo,
    ExecutionContext,
    SecurityContext,
    validate_input,
    validate_output,
    create_standard_input,
    create_standard_output,
    create_error_output,
    TracingContext,
    measure_performance
)

# Shared features - only import if available
try:
    from .prompt import (
        PromptManager,
        PromptTemplate,
        SystemPrompt,
        AgentPrompt,
        create_prompt_manager
    )
    PROMPT_AVAILABLE = True
except ImportError:
    PROMPT_AVAILABLE = False

try:
    from .rag import (
        RAGManager,
        Document,
        DocumentChunk,
        SearchResult,
        create_rag_manager
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from .memory import (
        MemoryManager,
        Memory,
        MemoryType,
        MemoryScope,
        create_memory_manager
    )
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from .knowledge import (
        KnowledgeManager,
        Entity,
        Relationship,
        Fact,
        create_knowledge_manager
    )
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from .tools import (
        ToolManager,
        Tool,
        ToolParameter,
        ToolExecution,
        create_tool_manager
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

try:
    from .context import (
        ContextManager,
        SessionState,
        ConversationHistory,
        UserPreferences,
        SharedContext,
        create_context_manager
    )
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False

# Base exports - always available
__all__ = [
    # Communication (always available)
    "StandardAgentInput",
    "StandardAgentOutput", 
    "ValidationError",
    "Priority",
    "MessageType",
    "Status",
    "TraceInfo",
    "ExecutionContext",
    "SecurityContext",
    "validate_input",
    "validate_output",
    "create_standard_input",
    "create_standard_output",
    "create_error_output",
    "TracingContext",
    "measure_performance"
]

# Add conditional exports
if PROMPT_AVAILABLE:
    __all__.extend([
        "PromptManager",
        "PromptTemplate",
        "SystemPrompt",
        "AgentPrompt",
        "create_prompt_manager"
    ])

if RAG_AVAILABLE:
    __all__.extend([
        "RAGManager",
        "Document",
        "DocumentChunk", 
        "SearchResult",
        "create_rag_manager"
    ])

if MEMORY_AVAILABLE:
    __all__.extend([
        "MemoryManager",
        "Memory",
        "MemoryType",
        "MemoryScope", 
        "create_memory_manager"
    ])

if KNOWLEDGE_AVAILABLE:
    __all__.extend([
        "KnowledgeManager",
        "Entity",
        "Relationship",
        "Fact",
        "create_knowledge_manager"
    ])

if TOOLS_AVAILABLE:
    __all__.extend([
        "ToolManager",
        "Tool",
        "ToolParameter",
        "ToolExecution",
        "create_tool_manager"
    ])

if CONTEXT_AVAILABLE:
    __all__.extend([
        "ContextManager",
        "SessionState",
        "ConversationHistory",
        "UserPreferences",
        "SharedContext",
        "create_context_manager"
    ])


class SharedServices:
    """
    Unified access point for all shared services.
    Provides a single interface to access all shared features.
    """
    
    def __init__(self):
        # Initialize with None - will be set during initialization
        self.prompt: 'PromptManager' = None
        self.rag: 'RAGManager' = None
        self.memory: 'MemoryManager' = None
        self.knowledge: 'KnowledgeManager' = None
        self.tools: 'ToolManager' = None
        self.context: 'ContextManager' = None
        self._initialized = False
        
        # Track what's available
        self.features_available = {
            "communication": True,  # Always available
            "prompt": PROMPT_AVAILABLE,
            "rag": RAG_AVAILABLE,
            "memory": MEMORY_AVAILABLE,
            "knowledge": KNOWLEDGE_AVAILABLE,
            "tools": TOOLS_AVAILABLE,
            "context": CONTEXT_AVAILABLE
        }
    
    async def initialize(self, config: dict = None):
        """Initialize all available shared services."""
        if self._initialized:
            return
        
        config = config or {}
        
        # Initialize available services
        if PROMPT_AVAILABLE:
            self.prompt = await create_prompt_manager(
                storage_path=config.get("prompt", {}).get("storage_path", ":memory:")
            )
        
        if RAG_AVAILABLE:
            self.rag = await create_rag_manager(
                storage_path=config.get("rag", {}).get("storage_path", ":memory:")
            )
        
        if MEMORY_AVAILABLE:
            self.memory = await create_memory_manager(
                storage_path=config.get("memory", {}).get("storage_path", ":memory:")
            )
        
        if KNOWLEDGE_AVAILABLE:
            self.knowledge = await create_knowledge_manager(
                storage_path=config.get("knowledge", {}).get("storage_path", ":memory:")
            )
        
        if TOOLS_AVAILABLE:
            self.tools = await create_tool_manager()
        
        if CONTEXT_AVAILABLE:
            self.context = await create_context_manager(
                storage_path=config.get("context", {}).get("storage_path", ":memory:")
            )
        
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown all initialized shared services."""
        if not self._initialized:
            return
        
        # Shutdown services in reverse order
        if self.context and CONTEXT_AVAILABLE:
            await self.context.shutdown()
        
        if self.tools and TOOLS_AVAILABLE:
            await self.tools.shutdown()
        
        if self.knowledge and KNOWLEDGE_AVAILABLE:
            await self.knowledge.shutdown()
        
        if self.memory and MEMORY_AVAILABLE:
            await self.memory.shutdown()
        
        if self.rag and RAG_AVAILABLE:
            await self.rag.shutdown()
        
        if self.prompt and PROMPT_AVAILABLE:
            await self.prompt.shutdown()
        
        self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if services are initialized."""
        return self._initialized
    
    def get_available_features(self) -> dict:
        """Get list of available features."""
        return self.features_available.copy()
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a specific feature is available."""
        return self.features_available.get(feature, False)


async def create_shared_services(config: dict = None) -> SharedServices:
    """
    Create and initialize a SharedServices instance.
    
    Args:
        config: Configuration dictionary for services
        
    Returns:
        Initialized SharedServices instance
    """
    services = SharedServices()
    await services.initialize(config)
    return services


def create_communication_system():
    """Create standardized communication utilities."""
    return {
        "input": StandardAgentInput,
        "output": StandardAgentOutput,
        "validate_input": validate_input,
        "validate_output": validate_output,
        "create_input": create_standard_input,
        "create_output": create_standard_output,
        "create_error": create_error_output,
        "tracing": TracingContext,
        "measure_performance": measure_performance
    }


def get_feature_status():
    """Get status of all shared features."""
    return {
        "communication": True,
        "prompt": PROMPT_AVAILABLE,
        "rag": RAG_AVAILABLE, 
        "memory": MEMORY_AVAILABLE,
        "knowledge": KNOWLEDGE_AVAILABLE,
        "tools": TOOLS_AVAILABLE,
        "context": CONTEXT_AVAILABLE
    }

# RAG System
from . import rag
from .rag import (
    RAGSystem,
    create_simple_rag_system,
    create_production_rag_system,
    # Core types
    VectorStoreType,
    EmbeddingModelType,
    DocumentType,
    RetrievalStrategy,
    Document,
    RAGQuery,
    RetrievalResult,
    RAGSystemConfig
)

# Memory System
from . import memory
from .memory import (
    MemorySystem,
    create_simple_memory_system,
    create_persistent_memory_system,
    create_distributed_memory_system,
    create_production_memory_system,
    # Core types
    MemoryType,
    MemoryStorageType,
    MemoryEntry,
    EpisodicMemory,
    SemanticMemory,
    WorkingMemory,
    ProceduralMemory,
    MemoryQuery,
    MemorySearchResult,
    MemorySystemConfig
)

# Placeholder imports for future shared features
# from . import knowledge
# from . import tools  
# from . import context
# from . import session

__all__ = [
    # Modules
    'rag',
    'memory',
    
    # RAG System
    'RAGSystem',
    'create_simple_rag_system', 
    'create_production_rag_system',
    'VectorStoreType',
    'EmbeddingModelType',
    'DocumentType',
    'RetrievalStrategy',
    'Document',
    'RAGQuery',
    'RetrievalResult',
    'RAGSystemConfig',
    
    # Memory System
    'MemorySystem',
    'create_simple_memory_system',
    'create_persistent_memory_system', 
    'create_distributed_memory_system',
    'create_production_memory_system',
    'MemoryType',
    'MemoryStorageType',
    'MemoryEntry',
    'EpisodicMemory',
    'SemanticMemory',
    'WorkingMemory',
    'ProceduralMemory',
    'MemoryQuery',
    'MemorySearchResult',
    'MemorySystemConfig'
]
