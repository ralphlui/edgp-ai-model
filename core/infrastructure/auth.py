"""
Authentication and authorization system for EDGP AI Model.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import logging
from enum import Enum

from ..config import settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Security
security = HTTPBearer()


class UserRole(str, Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class Permission(str, Enum):
    """System permissions."""
    # Agent permissions
    EXECUTE_AGENTS = "execute_agents"
    VIEW_AGENT_RESULTS = "view_agent_results"
    MANAGE_AGENTS = "manage_agents"
    
    # Data permissions
    VIEW_DATA_QUALITY = "view_data_quality"
    MANAGE_DATA_QUALITY = "manage_data_quality"
    VIEW_COMPLIANCE = "view_compliance"
    MANAGE_COMPLIANCE = "manage_compliance"
    
    # Remediation permissions
    VIEW_REMEDIATION = "view_remediation"
    CREATE_REMEDIATION = "create_remediation"
    EXECUTE_REMEDIATION = "execute_remediation"
    
    # Analytics permissions
    VIEW_ANALYTICS = "view_analytics"
    CREATE_DASHBOARDS = "create_dashboards"
    MANAGE_DASHBOARDS = "manage_dashboards"
    
    # System permissions
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_USERS = "manage_users"
    SYSTEM_CONFIG = "system_config"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.EXECUTE_AGENTS,
        Permission.VIEW_AGENT_RESULTS,
        Permission.MANAGE_AGENTS,
        Permission.VIEW_DATA_QUALITY,
        Permission.MANAGE_DATA_QUALITY,
        Permission.VIEW_COMPLIANCE,
        Permission.MANAGE_COMPLIANCE,
        Permission.VIEW_REMEDIATION,
        Permission.CREATE_REMEDIATION,
        Permission.EXECUTE_REMEDIATION,
        Permission.VIEW_ANALYTICS,
        Permission.CREATE_DASHBOARDS,
        Permission.MANAGE_DASHBOARDS,
        Permission.VIEW_AUDIT_LOGS,
        Permission.MANAGE_USERS,
        Permission.SYSTEM_CONFIG,
    ],
    UserRole.ANALYST: [
        Permission.EXECUTE_AGENTS,
        Permission.VIEW_AGENT_RESULTS,
        Permission.VIEW_DATA_QUALITY,
        Permission.MANAGE_DATA_QUALITY,
        Permission.VIEW_COMPLIANCE,
        Permission.MANAGE_COMPLIANCE,
        Permission.VIEW_REMEDIATION,
        Permission.CREATE_REMEDIATION,
        Permission.EXECUTE_REMEDIATION,
        Permission.VIEW_ANALYTICS,
        Permission.CREATE_DASHBOARDS,
    ],
    UserRole.VIEWER: [
        Permission.VIEW_AGENT_RESULTS,
        Permission.VIEW_DATA_QUALITY,
        Permission.VIEW_COMPLIANCE,
        Permission.VIEW_REMEDIATION,
        Permission.VIEW_ANALYTICS,
    ],
    UserRole.API_CLIENT: [
        Permission.EXECUTE_AGENTS,
        Permission.VIEW_AGENT_RESULTS,
        Permission.VIEW_DATA_QUALITY,
        Permission.VIEW_COMPLIANCE,
    ],
}


class TokenPayload:
    """JWT token payload structure."""
    
    def __init__(self, user_id: str, username: str, role: UserRole, permissions: List[Permission]):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.permissions = permissions


class AuthManager:
    """Main authentication manager."""
    
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating refresh token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create refresh token"
            )
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenPayload:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Extract token data
            user_id = payload.get("sub")
            username = payload.get("username")
            role = payload.get("role")
            
            if not user_id or not username or not role:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            # Get permissions for role
            permissions = ROLE_PERMISSIONS.get(UserRole(role), [])
            
            return TokenPayload(
                user_id=user_id,
                username=username,
                role=UserRole(role),
                permissions=permissions
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError as e:
            logger.error(f"JWT decode error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"edgp_{secrets.token_urlsafe(32)}"
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)


class User:
    """User model for authentication."""
    
    def __init__(self, user_id: str, username: str, email: str, role: UserRole, 
                 hashed_password: str, is_active: bool = True, api_key: Optional[str] = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.hashed_password = hashed_password
        self.is_active = is_active
        self.api_key = api_key
        self.created_at = datetime.utcnow()
        self.last_login = None
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        user_permissions = ROLE_PERMISSIONS.get(self.role, [])
        return permission in user_permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (without sensitive data)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


# In-memory user store (in production, this would be a database)
_users_store: Dict[str, User] = {}
_api_keys_store: Dict[str, str] = {}  # api_key -> user_id


class UserManager:
    """User management class."""
    
    def __init__(self):
        self.auth_manager = AuthManager()
        self._setup_default_users()
    
    def _setup_default_users(self):
        """Setup default admin user."""
        admin_user = User(
            user_id="admin-001",
            username="admin",
            email="admin@edgp.ai",
            role=UserRole.ADMIN,
            hashed_password=self.auth_manager.hash_password("admin123"),
            is_active=True
        )
        self.create_user(admin_user)
    
    def create_user(self, user: User) -> User:
        """Create a new user."""
        if user.username in [u.username for u in _users_store.values()]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        _users_store[user.user_id] = user
        
        # Generate API key if needed
        if user.role == UserRole.API_CLIENT:
            api_key = self.auth_manager.generate_api_key()
            user.api_key = api_key
            _api_keys_store[api_key] = user.user_id
        
        logger.info(f"Created user: {user.username} with role: {user.role}")
        return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in _users_store.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return _users_store.get(user_id)
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        user_id = _api_keys_store.get(api_key)
        if user_id:
            return _users_store.get(user_id)
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user = self.get_user_by_username(username)
        if not user:
            return None
        
        if not self.auth_manager.verify_password(password, user.hashed_password):
            return None
        
        if not user.is_active:
            return None
        
        user.last_login = datetime.utcnow()
        logger.info(f"User authenticated: {username}")
        return user
    
    def update_user_role(self, user_id: str, new_role: UserRole) -> Optional[User]:
        """Update user role."""
        user = _users_store.get(user_id)
        if not user:
            return None
        
        user.role = new_role
        logger.info(f"Updated user {user.username} role to: {new_role}")
        return user
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user."""
        user = _users_store.get(user_id)
        if not user:
            return False
        
        user.is_active = False
        logger.info(f"Deactivated user: {user.username}")
        return True


# Global instances
auth_manager = AuthManager()
user_manager = UserManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenPayload:
    """FastAPI dependency to get current authenticated user."""
    try:
        token = credentials.credentials
        return auth_manager.verify_token(token)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def get_current_user_with_api_key(request: Request) -> TokenPayload:
    """Alternative authentication via API key."""
    api_key = request.headers.get("X-API-Key")
    
    if api_key:
        user = user_manager.get_user_by_api_key(api_key)
        if user and user.is_active:
            permissions = ROLE_PERMISSIONS.get(user.role, [])
            return TokenPayload(
                user_id=user.user_id,
                username=user.username,
                role=user.role,
                permissions=permissions
            )
    
    # Fallback to JWT token
    credentials = await security(request)
    return await get_current_user(credentials)


def require_permissions(*required_permissions: Permission):
    """Decorator to require specific permissions."""
    def permission_checker(current_user: TokenPayload = Depends(get_current_user_with_api_key)):
        for permission in required_permissions:
            if permission not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission.value}"
                )
        return current_user
    return permission_checker


def require_role(*required_roles: UserRole):
    """Decorator to require specific roles."""
    def role_checker(current_user: TokenPayload = Depends(get_current_user_with_api_key)):
        if current_user.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role. Required: {[r.value for r in required_roles]}"
            )
        return current_user
    return role_checker


# Convenience functions for common permission checks
def require_admin():
    """Require admin role."""
    return require_role(UserRole.ADMIN)


def require_analyst_or_admin():
    """Require analyst or admin role."""
    return require_role(UserRole.ANALYST, UserRole.ADMIN)


def require_agent_execution():
    """Require agent execution permission."""
    return require_permissions(Permission.EXECUTE_AGENTS)


def require_data_quality_access():
    """Require data quality access permission."""
    return require_permissions(Permission.VIEW_DATA_QUALITY)


def require_compliance_access():
    """Require compliance access permission."""
    return require_permissions(Permission.VIEW_COMPLIANCE)
