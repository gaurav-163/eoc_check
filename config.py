"""
Configuration management for VAPI EOC Fetcher
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment"""
    
    # App Settings
    APP_NAME: str = "VAPI EOC Fetcher"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # VAPI Configuration
    VAPI_API_KEY: str
    VAPI_BASE_URL: str = "https://api.vapi.ai"
    VAPI_ORG_ID: Optional[str] = None
    
    # OpenAI (for analysis)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Database
    DATABASE_URL: str = "sqlite:///./vapi_calls.db"
    
    # Redis (optional caching)
    REDIS_URL: Optional[str] = None
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # Webhook
    WEBHOOK_SECRET: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()