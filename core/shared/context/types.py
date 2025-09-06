"""
Context Management Types

Comprehensive type system for managing AI agent context including conversation
history, session state, user preferences, and shared context across agents.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import json


class ContextType(str, Enum):
    """Types of context."""
    CONVERSATION = "conversation"
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    WORKFLOW = "workflow"
    TASK = "task"
    SYSTEM = "system"
    SHARED = "shared"


class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


class SessionStatus(str, Enum):
    """Session status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class ContextScope(str, Enum):
    """Context scope for sharing."""
    PRIVATE = "private"
    SESSION = "session"
    USER = "user"
    AGENT = "agent"
    GLOBAL = "global"


class Message(BaseModel):
    """Conversation message."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    agent_id: Optional[str] = Field(default=None, description="Agent ID if from agent")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls in message")
    function_call: Optional[Dict[str, Any]] = Field(default=None, description="Function call")
    
    # Context
    session_id: Optional[str] = Field(default=None, description="Session ID")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    thread_id: Optional[str] = Field(default=None, description="Thread ID")
    
    # Processing metadata
    tokens_used: Optional[int] = Field(default=None, description="Tokens used")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    model_used: Optional[str] = Field(default=None, description="Model used for generation")
    
    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    attachments: List[Dict[str, Any]] = Field(default_factory=list, description="Message attachments")
    
    class Config:
        schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the weather like today?",
                "timestamp": "2024-01-01T12:00:00Z",
                "metadata": {"intent": "weather_query"}
            }
        }


class ConversationHistory(BaseModel):
    """Conversation history container."""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Conversation ID")
    session_id: str = Field(..., description="Session ID")
    
    # Messages
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    message_count: int = Field(default=0, description="Total message count")
    
    # Metadata
    title: Optional[str] = Field(default=None, description="Conversation title")
    summary: Optional[str] = Field(default=None, description="Conversation summary")
    tags: List[str] = Field(default_factory=list, description="Conversation tags")
    
    # Participants
    user_id: Optional[str] = Field(default=None, description="User ID")
    agent_ids: List[str] = Field(default_factory=list, description="Participating agent IDs")
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now, description="Conversation start time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity time")
    ended_at: Optional[datetime] = Field(default=None, description="Conversation end time")
    
    # Statistics
    total_tokens: int = Field(default=0, description="Total tokens used")
    user_messages: int = Field(default=0, description="User message count")
    assistant_messages: int = Field(default=0, description="Assistant message count")
    
    def add_message(self, message: Message):
        """Add a message to the conversation."""
        message.conversation_id = self.conversation_id
        message.session_id = self.session_id
        
        self.messages.append(message)
        self.message_count += 1
        self.last_activity = datetime.now()
        
        # Update statistics
        if message.tokens_used:
            self.total_tokens += message.tokens_used
        
        if message.role == MessageRole.USER:
            self.user_messages += 1
        elif message.role == MessageRole.ASSISTANT:
            self.assistant_messages += 1
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages."""
        return self.messages[-count:] if count > 0 else self.messages
    
    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """Get messages by role."""
        return [msg for msg in self.messages if msg.role == role]
    
    def search_messages(self, query: str) -> List[Message]:
        """Search messages by content."""
        query_lower = query.lower()
        return [msg for msg in self.messages if query_lower in msg.content.lower()]


class UserPreferences(BaseModel):
    """User preferences and settings."""
    user_id: str = Field(..., description="User ID")
    
    # Communication preferences
    preferred_language: str = Field(default="en", description="Preferred language")
    communication_style: str = Field(default="professional", description="Communication style")
    detail_level: str = Field(default="medium", description="Detail level preference")
    
    # Interface preferences
    theme: str = Field(default="light", description="UI theme")
    timezone: str = Field(default="UTC", description="User timezone")
    date_format: str = Field(default="YYYY-MM-DD", description="Date format preference")
    
    # AI agent preferences
    preferred_agents: List[str] = Field(default_factory=list, description="Preferred agent IDs")
    agent_personalities: Dict[str, str] = Field(default_factory=dict, description="Agent personality preferences")
    
    # Privacy settings
    data_sharing: bool = Field(default=False, description="Allow data sharing")
    analytics_opt_in: bool = Field(default=True, description="Analytics opt-in")
    personalization: bool = Field(default=True, description="Enable personalization")
    
    # Custom settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom user settings")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def update_preference(self, key: str, value: Any):
        """Update a preference value."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.custom_settings[key] = value
        
        self.updated_at = datetime.now()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.custom_settings.get(key, default)


class SessionState(BaseModel):
    """Session state management."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session ID")
    user_id: Optional[str] = Field(default=None, description="User ID")
    
    # Session metadata
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    session_type: str = Field(default="user", description="Session type")
    
    # State data
    state_data: Dict[str, Any] = Field(default_factory=dict, description="Session state data")
    variables: Dict[str, Any] = Field(default_factory=dict, description="Session variables")
    flags: Dict[str, bool] = Field(default_factory=dict, description="Session flags")
    
    # Context
    current_conversation_id: Optional[str] = Field(default=None, description="Current conversation ID")
    active_agents: List[str] = Field(default_factory=list, description="Active agent IDs")
    current_workflow_id: Optional[str] = Field(default=None, description="Current workflow ID")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity time")
    expires_at: Optional[datetime] = Field(default=None, description="Session expiration time")
    
    # Statistics
    total_conversations: int = Field(default=0, description="Total conversations in session")
    total_messages: int = Field(default=0, description="Total messages in session")
    total_duration: float = Field(default=0.0, description="Total session duration in seconds")
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def set_variable(self, key: str, value: Any):
        """Set a session variable."""
        self.variables[key] = value
        self.update_activity()
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a session variable."""
        return self.variables.get(key, default)
    
    def set_flag(self, flag: str, value: bool = True):
        """Set a session flag."""
        self.flags[flag] = value
        self.update_activity()
    
    def has_flag(self, flag: str) -> bool:
        """Check if session has a flag set."""
        return self.flags.get(flag, False)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE and not self.is_expired()


class AgentContext(BaseModel):
    """Agent-specific context."""
    agent_id: str = Field(..., description="Agent ID")
    session_id: str = Field(..., description="Session ID")
    
    # Agent state
    current_task: Optional[str] = Field(default=None, description="Current task")
    task_history: List[str] = Field(default_factory=list, description="Task history")
    goals: List[str] = Field(default_factory=list, description="Agent goals")
    
    # Memory and knowledge
    working_memory: Dict[str, Any] = Field(default_factory=dict, description="Working memory")
    facts: List[str] = Field(default_factory=list, description="Known facts")
    assumptions: List[str] = Field(default_factory=list, description="Current assumptions")
    
    # Tool usage
    available_tools: List[str] = Field(default_factory=list, description="Available tool IDs")
    tool_usage_history: List[Dict[str, Any]] = Field(default_factory=list, description="Tool usage history")
    
    # Performance
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    error_history: List[Dict[str, Any]] = Field(default_factory=list, description="Error history")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def update_task(self, task: str):
        """Update current task."""
        if self.current_task:
            self.task_history.append(self.current_task)
        self.current_task = task
        self.updated_at = datetime.now()
    
    def add_fact(self, fact: str):
        """Add a known fact."""
        if fact not in self.facts:
            self.facts.append(fact)
            self.updated_at = datetime.now()
    
    def add_goal(self, goal: str):
        """Add an agent goal."""
        if goal not in self.goals:
            self.goals.append(goal)
            self.updated_at = datetime.now()
    
    def record_tool_usage(self, tool_id: str, success: bool, execution_time: float):
        """Record tool usage."""
        usage_record = {
            "tool_id": tool_id,
            "success": success,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        self.tool_usage_history.append(usage_record)
        self.updated_at = datetime.now()


class SharedContext(BaseModel):
    """Shared context across agents/sessions."""
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Context ID")
    name: str = Field(..., description="Context name")
    scope: ContextScope = Field(..., description="Context scope")
    
    # Context data
    data: Dict[str, Any] = Field(default_factory=dict, description="Shared data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context metadata")
    
    # Access control
    owner_id: Optional[str] = Field(default=None, description="Context owner ID")
    allowed_agents: List[str] = Field(default_factory=list, description="Allowed agent IDs")
    allowed_users: List[str] = Field(default_factory=list, description="Allowed user IDs")
    
    # Versioning
    version: int = Field(default=1, description="Context version")
    revision_history: List[Dict[str, Any]] = Field(default_factory=list, description="Revision history")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration time")
    
    def update_data(self, key: str, value: Any, agent_id: Optional[str] = None):
        """Update shared data."""
        old_value = self.data.get(key)
        self.data[key] = value
        
        # Record revision
        revision = {
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "version": self.version
        }
        self.revision_history.append(revision)
        
        self.version += 1
        self.updated_at = datetime.now()
    
    def can_access(self, agent_id: Optional[str] = None, user_id: Optional[str] = None) -> bool:
        """Check if agent/user can access this context."""
        if self.scope == ContextScope.GLOBAL:
            return True
        
        if agent_id and (not self.allowed_agents or agent_id in self.allowed_agents):
            return True
        
        if user_id and (not self.allowed_users or user_id in self.allowed_users):
            return True
        
        if agent_id == self.owner_id or user_id == self.owner_id:
            return True
        
        return False
    
    def is_expired(self) -> bool:
        """Check if context is expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False


class ContextConfig(BaseModel):
    """Context management configuration."""
    # Storage settings
    storage_path: str = Field(default="./data/context", description="Context storage path")
    message_retention_days: int = Field(default=90, description="Message retention period")
    session_timeout_hours: int = Field(default=24, description="Session timeout in hours")
    
    # Memory settings
    max_conversation_history: int = Field(default=1000, description="Max messages per conversation")
    max_context_size: int = Field(default=10000, description="Max context size in tokens")
    enable_compression: bool = Field(default=True, description="Enable context compression")
    
    # Sharing settings
    enable_shared_context: bool = Field(default=True, description="Enable shared context")
    default_context_scope: ContextScope = Field(default=ContextScope.SESSION, description="Default context scope")
    
    # Privacy settings
    enable_encryption: bool = Field(default=True, description="Enable context encryption")
    anonymize_data: bool = Field(default=False, description="Anonymize user data")
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable context caching")
    cache_size: int = Field(default=1000, description="Context cache size")
    enable_indexing: bool = Field(default=True, description="Enable context indexing")


class BaseContextStore(ABC):
    """Base class for context storage backends."""
    
    def __init__(self, config: ContextConfig):
        self.config = config
    
    @abstractmethod
    async def store_session(self, session: SessionState) -> bool:
        """Store session state."""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve session state."""
        pass
    
    @abstractmethod
    async def store_conversation(self, conversation: ConversationHistory) -> bool:
        """Store conversation history."""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Retrieve conversation history."""
        pass
    
    @abstractmethod
    async def store_user_preferences(self, preferences: UserPreferences) -> bool:
        """Store user preferences."""
        pass
    
    @abstractmethod
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Retrieve user preferences."""
        pass
    
    @abstractmethod
    async def store_shared_context(self, context: SharedContext) -> bool:
        """Store shared context."""
        pass
    
    @abstractmethod
    async def get_shared_context(self, context_id: str) -> Optional[SharedContext]:
        """Retrieve shared context."""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired data."""
        pass


class ContextAnalytics(BaseModel):
    """Analytics for context management."""
    total_sessions: int = Field(default=0, description="Total sessions")
    active_sessions: int = Field(default=0, description="Active sessions")
    total_conversations: int = Field(default=0, description="Total conversations")
    total_messages: int = Field(default=0, description="Total messages")
    
    # Usage patterns
    average_session_duration: float = Field(default=0.0, description="Average session duration")
    average_messages_per_conversation: float = Field(default=0.0, description="Average messages per conversation")
    most_active_users: List[Dict[str, Any]] = Field(default_factory=list, description="Most active users")
    
    # Context sharing
    shared_contexts: int = Field(default=0, description="Number of shared contexts")
    context_access_patterns: Dict[str, int] = Field(default_factory=dict, description="Context access patterns")
    
    # Performance
    storage_usage: Dict[str, int] = Field(default_factory=dict, description="Storage usage statistics")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate")
    
    # Time-based metrics
    sessions_today: int = Field(default=0, description="Sessions created today")
    messages_today: int = Field(default=0, description="Messages today")
    
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
