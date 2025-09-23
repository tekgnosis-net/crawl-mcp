"""
Configuration management for Crawl4AI MCP Server

This module handles loading and managing LLM configuration from MCP server settings.
"""

import json
import os
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Import our custom logging
from .utils.logging import MCPLogger


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider"""
    api_key: Optional[str]  # Direct API key value
    api_key_env: Optional[str]  # Environment variable name for API key
    base_url: Optional[str]
    base_url_env: Optional[str] = None  # Environment variable name for base URL (AOAI)
    api_version: Optional[str] = None  # API version for Azure OpenAI
    models: list = None
    extra_headers: Optional[Dict[str, str]] = None  # Additional HTTP headers for API requests
    custom_chat_llm_api_key: Optional[str] = None  # Custom chat LLM API key (if different)
    custom_chat_llm_base_url: Optional[str] = None  # Custom chat LLM base URL (if different)
    custom_chat_llm_model: Optional[str] = None  # Custom chat LLM model name
    custom_embeddings_llm_api_key: Optional[str] = None  # Custom embeddings LLM API key (if different)
    custom_embeddings_llm_base_url: Optional[str] = None  # Custom embeddings LLM base URL (if different)
    custom_embeddings_llm_model: Optional[str] = None  # Custom embeddings LLM model name
    use_custom_endpoints: bool = False  # Whether to use custom endpoints for chat/embeddings
    logging_enabled: bool = True  # Enable detailed logging
    logging_level: str = "INFO"  # Logging level (e.g., INFO, DEBUG)
    custom_logging_path: Optional[str] = None  # Custom logging path (if different)


@dataclass
class MCPLLMConfig:
    """Complete LLM configuration from MCP settings"""
    default_provider: str
    default_model: str
    providers: Dict[str, LLMProviderConfig]


class ConfigManager:
    """Manages loading and accessing LLM configuration from MCP settings"""
    self.logging_enabled: bool = True
    self.logging_level: str = "INFO"
    self.custom_logging_path: Optional[str] = None
    self.logger: MCPLogger = None
    self.llm_config: Optional[MCPLLMConfig] = None

    def __init__(self):
        self.llm_config: Optional[MCPLLMConfig] = None
        self._load_env_vars()
        self._load_config()
    
    def _load_env_vars(self):
        """Load environment variables from .env file if available"""
        if DOTENV_AVAILABLE:
            # Get module directory and construct absolute paths
            module_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(module_dir, '..')
            
            # Look for .env file with absolute paths
            env_paths = [
                os.path.join(project_root, '.env'),  # Project root
                '.env',  # Current directory (fallback)
                os.path.join(os.getcwd(), '.env')  # Working directory
            ]
            
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path, override=False)  # Don't override existing env vars
                    print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
                    break
            else:
                # Try to load from any .env file in the working directory
                try:
                    load_dotenv(override=False)
                except:
                    pass  # Silently fail if no .env file found
        else:
            print("⚠️ python-dotenv not available. Install with: pip install python-dotenv", file=sys.stderr)
    
    def _load_config(self):
        """Load LLM configuration from MCP server environment or config files"""
        
        # Setup logging level from environment variable
        log_level = os.getenv('MCP_LOGGING_LEVEL', 'INFO').upper()
        if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            log_level = 'INFO'
            print(f"ℹ️ MCP Logging Level: {log_level}", file=sys.stderr)
        
        log_file = os.getenv('MCP_LOG_FILE', None)
        if log_file:
            print(f"ℹ️ MCP Log File: {log_file}", file=sys.stderr)
            self.logger = MCPLogger(level=log_level, log_file=log_file)
        else:
            print(f"ℹ️ MCP Logging to console at level: {log_level}", file=sys.stderr)
            self.logger = MCPLogger(level=log_level)
        # Try to load from environment variable (if MCP passes config)
        config_json = os.getenv('MCP_LLM_CONFIG')
        if config_json:
            try:
                config_data = json.loads(config_json)
                self.llm_config = self._parse_llm_config(config_data)
                return
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse MCP_LLM_CONFIG: {e}", file=sys.stderr)
        
        # Get module directory and construct absolute paths
        module_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(module_dir, '..')
        
        # Try to load from claude_desktop_config.json with absolute paths
        config_files = [
            os.path.join(project_root, 'claude_desktop_config.json'),  # Project root
            'claude_desktop_config.json',  # Current directory (fallback)
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        desktop_config = json.load(f)
                    
                    # Extract crawl4ai server config
                    mcp_servers = desktop_config.get('mcpServers', {})
                    crawl4ai_config = mcp_servers.get('crawl4ai', {})
                    llm_config_data = crawl4ai_config.get('llm_config')
                    
                    if llm_config_data:
                        self.llm_config = self._parse_llm_config(llm_config_data)
                        print(f"✅ Loaded LLM config from {config_file}", file=sys.stderr)
                        
                        # Log API key status for each provider
                        for provider_name, provider_config in self.llm_config.providers.items():
                            if provider_config.api_key:
                                print(f"✅ Found direct API key for {provider_name} (starts with: {provider_config.api_key[:15]}...)", file=sys.stderr)
                            elif provider_config.api_key_env:
                                env_value = os.getenv(provider_config.api_key_env)
                                if env_value:
                                    print(f"✅ Found environment API key for {provider_name} from {provider_config.api_key_env} (starts with: {env_value[:15]}...)", file=sys.stderr)
                                else:
                                    print(f"❌ Environment variable {provider_config.api_key_env} not set for {provider_name}", file=sys.stderr)
                            else:
                                print(f"❌ No API key configuration for {provider_name}", file=sys.stderr)
                        return
                        
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Warning: Failed to load config from {config_file}: {e}", file=sys.stderr)
                    continue
        
        if not self.llm_config:
            print("⚠️ No LLM configuration found in MCP_LLM_CONFIG or claude_desktop_config.json", file=sys.stderr)
            print("Will check for custom LLM environment variables or use defaults.", file=sys.stderr)
            if os.getenv('USE_CUSTOM_OPENAI_ENDPOINTS', 'false').lower() == 'true':
                print("✅ Detected custom OpenAI endpoints from environment variables.", file=sys.stderr)
                self.llm_config = MCPLLMConfig(
                    default_provider=os.getenv('CUSTOM_OPENAI_PROVIDER', 'localai'),
                    default_model=os.getenv('CUSTOM_OPENAI_CHAT_MODEL_NAME', 'gpt-4.1'),
                    providers={
                        default_provider: LLMProviderConfig(
                            custom_chat_llm_api_key=os.getenv('CUSTOM_OPENAI_CHAT_API_KEY'),
                            custom_chat_llm_base_url=os.getenv('CUSTOM_OPENAI_CHAT_BASE_URL'),
                            custom_chat_llm_model=os.getenv('CUSTOM_OPENAI_CHAT_MODEL_NAME', 'gpt-4.1'),
                            custom_embeddings_llm_api_key=os.getenv('CUSTOM_OPENAI_EMBEDDINGS_API_KEY'),
                            custom_embeddings_llm_base_url=os.getenv('CUSTOM_OPENAI_EMBEDDINGS_BASE_URL'),
                            custom_embeddings_llm_model=os.getenv('CUSTOM_OPENAI_EMBEDDINGS_MODEL_NAME', 'text-embedding-3-small'),
                            use_custom_endpoints=True,
                            api_key_env='CUSTOM_OPENAI_PROVIDER',
                            base_url='CUSTOM_OPENAI_CHAT_BASE_URL',
                            api_key='CUSTOM_OPENAI_CHAT_API_KEY',
                            api_version='v1',
                            models=[os.getenv('CUSTOM_OPENAI_CHAT_MODEL_NAME', 'gpt-4.1')]
                        )
                    }
                )
                print("✅ Loaded custom OpenAI LLM configuration from environment variables.", file=sys.stderr)
            return

        # Fallback to default configuration
        print("Using default LLM configuration", file=sys.stderr)
        self.llm_config = self._get_default_config()
    
    def _parse_llm_config(self, config_data: Dict[str, Any]) -> MCPLLMConfig:
        """Parse LLM configuration from JSON data"""
        providers = {}
        for name, provider_data in config_data.get('providers', {}).items():
            providers[name] = LLMProviderConfig(
                api_key=provider_data.get('api_key'),  # Direct API key
                api_key_env=provider_data.get('api_key_env'),  # Environment variable name
                base_url=provider_data.get('base_url'),
                base_url_env=provider_data.get('base_url_env'),  # Environment variable for base URL
                api_version=provider_data.get('api_version'),  # API version for Azure
                models=provider_data.get('models', []),
                extra_headers=provider_data.get('extra_headers')  # Custom HTTP headers
            )
        
        return MCPLLMConfig(
            default_provider=config_data.get('default_provider', 'openai'),
            default_model=config_data.get('default_model', 'gpt-4.1'),
            providers=providers
        )
    
    def _get_default_config(self) -> MCPLLMConfig:
        """Get default LLM configuration when no config is found"""
        return MCPLLMConfig(
            default_provider='openai',
            default_model='gpt-4.1',
            providers={
                'openai': LLMProviderConfig(
                    api_key=None,  # No direct API key by default
                    api_key_env='OPENAI_API_KEY',
                    base_url=None,
                    models=['gpt-4.1', 'gpt-o4-mini']
                ),
                'anthropic': LLMProviderConfig(
                    api_key=None,  # No direct API key by default
                    api_key_env='ANTHROPIC_API_KEY',
                    base_url=None,
                    models=['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307']
                ),
                'ollama': LLMProviderConfig(
                    api_key=None,  # No API key needed for Ollama
                    api_key_env=None,
                    base_url='http://localhost:11434',
                    models=['llama3.3', 'qwen2.5']
                ),
                'aoai': LLMProviderConfig(
                    api_key=None,
                    api_key_env='AZURE_OPENAI_API_KEY',
                    base_url=None,
                    base_url_env='AZURE_OPENAI_ENDPOINT',
                    api_version='2025-04-01-preview',
                    models=['gpt-4.1', 'gpt-o4-mini']
                ),
                'localai': LLMProviderConfig(
                    api_key=None,
                    api_key_env='LOCALAI_API_KEY',  # Optional API key for LocalAI
                    base_url=None,
                    base_url_env='LOCALAI_BASE_URL',  # e.g., http://localhost:8080
                    models=['/models/OpenAI-20B-NEO-CODE2-Plus-Uncensored-IQ4_NL.gguf', 'Mistral-7B-Instruct-v0.1.gguf']
                )
            }
        )
    
    def get_provider_config(self, provider: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific provider"""
        if not self.llm_config:
            return None
        return self.llm_config.providers.get(provider)
    
    def get_default_provider(self) -> str:
        """Get the default LLM provider"""
        if not self.llm_config:
            return 'openai'
        return self.llm_config.default_provider
    
    def get_default_model(self) -> str:
        """Get the default LLM model"""
        if not self.llm_config:
            return 'gpt-4.1'
        return self.llm_config.default_model
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider (direct key or from environment variables)"""
        provider_config = self.get_provider_config(provider)
        if not provider_config:
            print(f"Warning: No config found for provider {provider}", file=sys.stderr)
            return None
        
        # First try custom llm if configured
        if provider_config.use_custom_endpoints and provider_config.custom_chat_llm_api_key:
            print(f"✅ Loaded custom chat LLM API key for {provider} (starts with: {provider_config.custom_chat_llm_api_key[:15]}...)", file=sys.stderr)
            return provider_config.custom_chat_llm_api_key

        # Next, try direct API key
        if provider_config.api_key:
            print(f"✅ Loaded API key for {provider} from configuration (starts with: {provider_config.api_key[:15]}...)", file=sys.stderr)
            return provider_config.api_key
        
        # Then, try environment variable
        if provider_config.api_key_env:
            env_key = os.getenv(provider_config.api_key_env)
            if env_key:
                print(f"✅ Loaded API key for {provider} from environment variable {provider_config.api_key_env} (starts with: {env_key[:15]}...)", file=sys.stderr)
                return env_key
            else:
                print(f"❌ Environment variable {provider_config.api_key_env} not set for {provider}", file=sys.stderr)
        
        print(f"Warning: No API key found for {provider}", file=sys.stderr)
        return None
    
    def get_base_url(self, provider: str) -> Optional[str]:
        """Get base URL for a provider (direct URL or from environment variable)"""
        provider_config = self.get_provider_config(provider)
        if not provider_config:
            return None
        
        # First try custom llm if configured
        if provider_config.use_custom_endpoints and provider_config.custom_chat_llm_base_url:
            print(f"✅ Loaded custom chat LLM base URL for {provider} from configuration: {provider_config.custom_chat_llm_base_url}", file=sys.stderr)
            return provider_config.custom_chat_llm_base_url

        # Next, try direct base URL
        if provider_config.base_url:
            return provider_config.base_url
        
        # Then, try environment variable (especially for AOAI)
        if provider_config.base_url_env:
            env_url = os.getenv(provider_config.base_url_env)
            if env_url:
                print(f"✅ Loaded base URL for {provider} from environment variable {provider_config.base_url_env}", file=sys.stderr)
                return env_url
            else:
                print(f"❌ Environment variable {provider_config.base_url_env} not set for {provider}", file=sys.stderr)
        
        return None
    
    def get_extra_headers(self, provider: str) -> Optional[Dict[str, str]]:
        """Get extra headers for a provider"""
        provider_config = self.get_provider_config(provider)
        if not provider_config:
            return None
        return provider_config.extra_headers
    
    def has_valid_api_key(self, provider: str) -> bool:
        """Check if a provider has a valid API key available"""
        provider_config = self.get_provider_config(provider)
        if not provider_config:
            return False
        
        # First check custom llm if configured
        if provider_config.use_custom_endpoints and provider_config.custom_chat_llm_api_key:
            return True
        elseif provider_config.use_custom_endpoints and provider_config.custom_chat_llm_base_url:
            # If no API key but custom base URL is set, assume it's valid (e.g., LocalAI without key)
            return True

        # Check direct API key
        if provider_config.api_key:
            return True
        
        # Check environment variable
        if provider_config.api_key_env:
            env_key = os.getenv(provider_config.api_key_env)
            if env_key:
                return True
        
        # Ollama doesn't need API key
        if provider == 'ollama':
            return True
        
        return False
    
    def get_available_providers(self) -> list:
        """Get list of providers with valid API keys"""
        if not self.llm_config:
            return []
        
        available = []
        for provider in self.llm_config.providers.keys():
            if self.has_valid_api_key(provider):
                available.append(provider)
        
        return available
    
    def create_llm_config(self, provider: Optional[str] = None, model: Optional[str] = None):
        """Create a Crawl4AI LLMConfig object with the specified or default provider/model
        
        If the specified provider doesn't have a valid API key, it will try other providers
        in order of preference: openai -> aoai -> anthropic -> ollama
        """
        from crawl4ai import LLMConfig
        
        # Define fallback order
        fallback_order = ['openai', 'aoai', 'anthropic', 'ollama']
        
        # If provider is specified, try it first
        if provider:
            if self.llm_config.get_provider_config(provider).use_custom_endpoints:
                print(f"ℹ️ Using custom endpoints for provider: {provider}", file=sys.stderr)

            if self.has_valid_api_key(provider):
                target_provider = provider
                target_model = model or self.get_default_model()
                print(f"✅ Using specified provider: {provider}", file=sys.stderr)
            else:
                print(f"⚠️ Specified provider {provider} has no valid API key, trying fallback providers...", file=sys.stderr)
                target_provider = None
        else:
            # Start with default provider
            default_provider = self.get_default_provider()
            if self.has_valid_api_key(default_provider):
                target_provider = default_provider
                target_model = model or self.get_default_model()
                print(f"✅ Using default provider: {default_provider}", file=sys.stderr)
            else:
                print(f"⚠️ Default provider {default_provider} has no valid API key, trying fallback providers...", file=sys.stderr)
                target_provider = None
        
        # If no valid provider found yet, try fallback order
        if not target_provider:
            for fallback_provider in fallback_order:
                if self.has_valid_api_key(fallback_provider):
                    target_provider = fallback_provider
                    # Use compatible model from the working provider
                    target_model = model or self._get_compatible_model(fallback_provider)
                    print(f"✅ Using fallback provider: {fallback_provider}", file=sys.stderr)
                    break
        
        # If still no valid provider found, raise error
        if not target_provider:
            available_providers = self.get_available_providers()
            if available_providers:
                raise ValueError(f"No valid API key found for specified provider. Available providers: {available_providers}")
            else:
                raise ValueError("No providers with valid API keys found. Please configure at least one provider.")
        
        # Get provider configuration
        provider_config = self.get_provider_config(target_provider)
        if not provider_config:
            raise ValueError(f"Unknown provider: {target_provider}")
        
        # Validate model is supported by the provider
        if target_model not in provider_config.models:
            print(f"⚠️ Model {target_model} not supported by {target_provider}, using first available model", file=sys.stderr)
            target_model = provider_config.models[0] if provider_config.models else 'default'
        
        # Get API key, base URL, and extra headers
        api_token = self.get_api_key(target_provider)
        base_url = self.get_base_url(target_provider)
        extra_headers = self.get_extra_headers(target_provider)
        
        print(f"🚀 Creating LLM config: {target_provider}/{target_model}", file=sys.stderr)
        
        # Create LLMConfig with extra_headers support
        config_params = {
            "provider": f"{target_provider}/{target_model}",
            "api_token": api_token,
            "base_url": base_url
        }
        
        # Add extra_headers if available and supported by current Crawl4AI version
        if extra_headers:
            try:
                llm_config = LLMConfig(**config_params, extra_headers=extra_headers)
                print(f"✅ Applied extra headers for {target_provider}: {list(extra_headers.keys())}", file=sys.stderr)
            except TypeError:
                # Fallback for older Crawl4AI versions without extra_headers support
                llm_config = LLMConfig(**config_params)
                print(f"⚠️ Extra headers specified but not supported by current Crawl4AI version", file=sys.stderr)
        else:
            llm_config = LLMConfig(**config_params)
        
        return llm_config
    
    def _get_compatible_model(self, provider: str) -> str:
        """Get a compatible model for the given provider"""
        provider_config = self.get_provider_config(provider)
        if not provider_config or not provider_config.models:
            return 'default'
        
        # Return the first model as default
        return provider_config.models[0]
    
    def validate_provider_model(self, provider: str, model: str) -> bool:
        """Validate if a provider/model combination is supported"""
        provider_config = self.get_provider_config(provider)
        if not provider_config:
            return False
        return model in provider_config.models
    
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, list]:
        """List all available models, optionally filtered by provider"""
        if not self.llm_config:
            return {}
        
        if provider:
            provider_config = self.get_provider_config(provider)
            return {provider: provider_config.models if provider_config else []}
        
        return {name: config.models for name, config in self.llm_config.providers.items()}


# Global configuration manager instance
config_manager = ConfigManager()


def get_llm_config(provider: Optional[str] = None, model: Optional[str] = None):
    """Convenience function to get LLMConfig with provider/model"""
    return config_manager.create_llm_config(provider, model)


def get_default_provider() -> str:
    """Convenience function to get default provider"""
    return config_manager.get_default_provider()


def get_default_model() -> str:
    """Convenience function to get default model"""
    return config_manager.get_default_model()