"""
Authentication endpoints for EDGP AI Model.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

from ..auth import (
    auth_manager, user_manager, User, UserRole, Permission,
    get_current_user, require_admin, require_permissions,
    TokenPayload
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBasic()


# Pydantic models for requests/responses
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class CreateUserRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: UserRole


class UpdateUserRoleRequest(BaseModel):
    role: UserRole


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str
    last_login: Optional[str] = None
    api_key: Optional[str] = None


class ApiKeyResponse(BaseModel):
    api_key: str
    user_id: str
    username: str


@router.post("/login", response_model=TokenResponse)
async def login(login_request: LoginRequest):
    """
    Authenticate user and return JWT tokens.
    """
    try:
        user = user_manager.authenticate_user(
            login_request.username, 
            login_request.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Create token payload
        token_data = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value
        }
        
        # Create tokens
        access_token = auth_manager.create_access_token(token_data)
        refresh_token = auth_manager.create_refresh_token(token_data)
        
        logger.info(f"User logged in: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=auth_manager.access_token_expire_minutes * 60,
            user=user.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    """
    try:
        # Verify refresh token
        token_payload = auth_manager.verify_token(
            refresh_request.refresh_token, 
            token_type="refresh"
        )
        
        # Get user details
        user = user_manager.get_user_by_id(token_payload.user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        token_data = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value
        }
        
        access_token = auth_manager.create_access_token(token_data)
        new_refresh_token = auth_manager.create_refresh_token(token_data)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=auth_manager.access_token_expire_minutes * 60,
            user=user.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/users", response_model=UserResponse, dependencies=[Depends(require_admin())])
async def create_user(
    user_request: CreateUserRequest,
    current_user: TokenPayload = Depends(require_admin())
):
    """
    Create a new user (admin only).
    """
    try:
        # Check if username already exists
        existing_user = user_manager.get_user_by_username(user_request.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Create new user
        new_user = User(
            user_id=f"user-{datetime.utcnow().timestamp()}",
            username=user_request.username,
            email=user_request.email,
            role=user_request.role,
            hashed_password=auth_manager.hash_password(user_request.password)
        )
        
        created_user = user_manager.create_user(new_user)
        
        logger.info(f"User created by {current_user.username}: {created_user.username}")
        
        response_data = created_user.to_dict()
        if created_user.api_key:
            response_data["api_key"] = created_user.api_key
        
        return UserResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/users", response_model=List[UserResponse], dependencies=[Depends(require_admin())])
async def list_users(current_user: TokenPayload = Depends(require_admin())):
    """
    List all users (admin only).
    """
    try:
        users = list(user_manager._users_store.values())
        return [UserResponse(**user.to_dict()) for user in users]
        
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(require_admin())])
async def get_user(
    user_id: str,
    current_user: TokenPayload = Depends(require_admin())
):
    """
    Get user by ID (admin only).
    """
    user = user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    response_data = user.to_dict()
    if user.api_key:
        response_data["api_key"] = user.api_key
    
    return UserResponse(**response_data)


@router.put("/users/{user_id}/role", response_model=UserResponse, dependencies=[Depends(require_admin())])
async def update_user_role(
    user_id: str,
    role_request: UpdateUserRoleRequest,
    current_user: TokenPayload = Depends(require_admin())
):
    """
    Update user role (admin only).
    """
    try:
        updated_user = user_manager.update_user_role(user_id, role_request.role)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User role updated by {current_user.username}: {updated_user.username} -> {role_request.role}")
        
        return UserResponse(**updated_user.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user role error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user role"
        )


@router.post("/users/{user_id}/deactivate", dependencies=[Depends(require_admin())])
async def deactivate_user(
    user_id: str,
    current_user: TokenPayload = Depends(require_admin())
):
    """
    Deactivate user (admin only).
    """
    try:
        success = user_manager.deactivate_user(user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user = user_manager.get_user_by_id(user_id)
        logger.info(f"User deactivated by {current_user.username}: {user.username}")
        
        return {"message": "User deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deactivate user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate user"
        )


@router.post("/api-key", response_model=ApiKeyResponse, dependencies=[Depends(require_admin())])
async def generate_api_key(
    username: str,
    current_user: TokenPayload = Depends(require_admin())
):
    """
    Generate API key for a user (admin only).
    """
    try:
        user = user_manager.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate new API key
        api_key = auth_manager.generate_api_key()
        user.api_key = api_key
        user_manager._api_keys_store[api_key] = user.user_id
        
        logger.info(f"API key generated by {current_user.username} for user: {username}")
        
        return ApiKeyResponse(
            api_key=api_key,
            user_id=user.user_id,
            username=user.username
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate API key"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: TokenPayload = Depends(get_current_user)):
    """
    Get current user profile.
    """
    user = user_manager.get_user_by_id(current_user.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(**user.to_dict())


@router.get("/permissions")
async def get_current_user_permissions(current_user: TokenPayload = Depends(get_current_user)):
    """
    Get current user's permissions.
    """
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "role": current_user.role.value,
        "permissions": [perm.value for perm in current_user.permissions]
    }


@router.get("/roles")
async def get_available_roles(current_user: TokenPayload = Depends(require_admin())):
    """
    Get available user roles (admin only).
    """
    return {
        "roles": [
            {
                "value": role.value,
                "permissions": [perm.value for perm in permissions]
            }
            for role, permissions in {
                UserRole.ADMIN: [Permission.MANAGE_USERS, Permission.SYSTEM_CONFIG],
                UserRole.ANALYST: [Permission.EXECUTE_AGENTS, Permission.MANAGE_DATA_QUALITY],
                UserRole.VIEWER: [Permission.VIEW_AGENT_RESULTS, Permission.VIEW_ANALYTICS],
                UserRole.API_CLIENT: [Permission.EXECUTE_AGENTS, Permission.VIEW_AGENT_RESULTS]
            }.items()
        ]
    }


@router.post("/logout")
async def logout(current_user: TokenPayload = Depends(get_current_user)):
    """
    Logout user (client-side token removal).
    """
    # In a production system, you might want to maintain a token blacklist
    # For now, we'll just return success and let the client handle token removal
    logger.info(f"User logged out: {current_user.username}")
    return {"message": "Logged out successfully"}
