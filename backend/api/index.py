"""
Vercel Serverless Function Entry Point for FastAPI application.
Exposes the FastAPI instance for Vercel deployment.
"""
from api.app import app

# Export ASGI application for Vercel Serverless Functions
__all__ = ["app"]
