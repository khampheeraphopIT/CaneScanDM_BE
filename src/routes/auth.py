from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, field_validator
from src.config.database import get_db
from src.models.user import User
from src.utils.auth import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)


# ============ Schemas ============
class RegisterRequest(BaseModel):
    phone: str
    name: str
    password: str

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 10 or not v.isdigit():
            raise ValueError('เบอร์โทรศัพท์ไม่ถูกต้อง')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 4:
            raise ValueError('รหัสผ่านต้องมีอย่างน้อย 4 ตัวอักษร')
        return v


class LoginRequest(BaseModel):
    phone: str
    password: str


class AuthResponse(BaseModel):
    success: bool
    message: str
    token: str | None = None
    user: dict | None = None


class UserResponse(BaseModel):
    id: int
    phone: str
    name: str


# ============ Dependency ============
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User | None:
    """Get current user from JWT token (optional - returns None if not authenticated)"""
    if not credentials:
        return None
    
    payload = decode_access_token(credentials.credentials)
    if not payload:
        return None
    
    user_id = int(payload.get("sub", 0))
    if not user_id:
        return None
    
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


# ============ Endpoints ============
@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    # Check if phone already exists
    existing = await db.execute(select(User).where(User.phone == request.phone))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="เบอร์โทรศัพท์นี้ถูกใช้งานแล้ว"
        )
    
    # Create new user
    new_user = User(
        phone=request.phone,
        name=request.name,
        password_hash=hash_password(request.password)
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Generate token
    token = create_access_token(new_user.id, new_user.phone)
    
    return AuthResponse(
        success=True,
        message="สมัครสมาชิกสำเร็จ",
        token=token,
        user={"id": new_user.id, "phone": new_user.phone, "name": new_user.name}
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Login with phone and password"""
    # Find user
    result = await db.execute(select(User).where(User.phone == request.phone))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="เบอร์โทรศัพท์หรือรหัสผ่านไม่ถูกต้อง"
        )
    
    # Generate token
    token = create_access_token(user.id, user.phone)
    
    return AuthResponse(
        success=True,
        message="เข้าสู่ระบบสำเร็จ",
        token=token,
        user={"id": user.id, "phone": user.phone, "name": user.name}
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get current user info (requires authentication)"""
    if not credentials:
        raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบ")
    
    user = await get_current_user(credentials, db)
    if not user:
        raise HTTPException(status_code=401, detail="Token ไม่ถูกต้องหรือหมดอายุ")
    
    return UserResponse(id=user.id, phone=user.phone, name=user.name)
