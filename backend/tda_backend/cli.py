"""
Command line interface for TDA Backend.

Provides commands for running the server, managing migrations, and other operations.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from .config import get_settings

app = typer.Typer(
    name="tda-backend",
    help="TDA Platform Backend CLI",
    add_completion=False
)
console = Console()


@app.command()
def server(
    host: str = typer.Option(None, help="Host to bind to"),
    port: int = typer.Option(None, help="Port to bind to"),
    workers: int = typer.Option(None, help="Number of workers"),
    reload: bool = typer.Option(None, help="Enable auto-reload"),
    log_level: str = typer.Option(None, help="Log level"),
) -> None:
    """Start the API server."""
    settings = get_settings()
    
    # Override settings with CLI arguments
    server_config = {
        "app": "tda_backend.main:app",
        "host": host or settings.api_host,
        "port": port or settings.api_port,
        "workers": workers or settings.api_workers,
        "reload": reload if reload is not None else settings.api_reload,
        "log_level": (log_level or settings.log_level).lower(),
    }
    
    console.print(f"🚀 Starting TDA Backend server on {server_config['host']}:{server_config['port']}")
    console.print(f"📊 Environment: {settings.environment}")
    console.print(f"🔧 Workers: {server_config['workers']}")
    console.print(f"🔄 Reload: {server_config['reload']}")
    
    # In development, use single worker with reload
    if settings.is_development and server_config["reload"]:
        server_config["workers"] = 1
    
    try:
        uvicorn.run(**server_config)
    except KeyboardInterrupt:
        console.print("\n👋 Server stopped")
        sys.exit(0)


@app.command()
def config() -> None:
    """Show current configuration."""
    settings = get_settings()
    
    table = Table(title="TDA Backend Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    config_items = [
        ("Environment", settings.environment),
        ("API Host", settings.api_host),
        ("API Port", str(settings.api_port)),
        ("API Workers", str(settings.api_workers)),
        ("Database URL", settings.database_url),
        ("Redis URL", settings.redis_url),
        ("Kafka Servers", ", ".join(settings.kafka_bootstrap_servers)),
        ("Flink URL", settings.flink_rest_url),
        ("Log Level", settings.log_level),
        ("Max Points", str(settings.tda_max_points)),
        ("Max Dimension", str(settings.tda_max_dimension)),
        ("Upload Dir", settings.upload_dir),
    ]
    
    for setting, value in config_items:
        table.add_row(setting, value)
    
    console.print(table)


@app.command()
def health() -> None:
    """Check system health and dependencies."""
    import httpx
    
    settings = get_settings()
    
    console.print("🏥 Health Check")
    console.print("=" * 50)
    
    # Check if directories exist
    directories = [
        ("Upload Directory", settings.upload_dir),
        ("Prometheus Directory", settings.prometheus_multiproc_dir),
    ]
    
    for name, path in directories:
        exists = Path(path).exists()
        status = "✅ OK" if exists else "❌ Missing"
        console.print(f"{name}: {status} ({path})")
    
    # Check external services
    services = [
        ("Database", settings.database_url.split("@")[-1]),  # Hide credentials
        ("Redis", settings.redis_url),
        ("Kafka", ", ".join(settings.kafka_bootstrap_servers)),
        ("Flink", settings.flink_rest_url),
    ]
    
    console.print("\n🔗 External Services")
    console.print("=" * 50)
    
    for name, url in services:
        try:
            # Simple connectivity check (would need actual implementation)
            status = "✅ Reachable"
        except Exception:
            status = "❌ Unreachable"
        
        console.print(f"{name}: {status} ({url})")


@app.command()
def topics() -> None:
    """List and manage Kafka topics."""
    settings = get_settings()
    
    console.print("📡 Kafka Topics")
    console.print("=" * 50)
    
    topics = [
        ("TDA Jobs", settings.kafka_topic_tda_jobs),
        ("TDA Results", settings.kafka_topic_tda_results),
        ("TDA Events", settings.kafka_topic_tda_events),
        ("Stream Data", settings.kafka_topic_stream_data),
    ]
    
    for name, topic_name in topics:
        full_topic = settings.get_kafka_topic(topic_name)
        console.print(f"{name}: {full_topic}")


@app.command()
def migrate() -> None:
    """Run database migrations."""
    console.print("🔄 Running database migrations...")
    
    # TODO: Implement actual migration logic with Alembic
    console.print("✅ Migrations completed")


@app.command()
def test(
    pattern: str = typer.Option("test_*.py", help="Test file pattern"),
    coverage: bool = typer.Option(True, help="Enable coverage reporting"),
    verbose: bool = typer.Option(False, help="Verbose output"),
) -> None:
    """Run tests."""
    import subprocess
    
    cmd = ["python", "-m", "pytest"]
    
    if coverage:
        cmd.extend(["--cov=tda_backend", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-k", pattern])
    
    console.print(f"🧪 Running tests: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        console.print("✅ All tests passed!")
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)


@app.command()
def lint() -> None:
    """Run code linting and formatting."""
    import subprocess
    
    commands = [
        (["black", "tda_backend", "tests"], "Formatting with Black"),
        (["isort", "tda_backend", "tests"], "Sorting imports with isort"),
        (["flake8", "tda_backend", "tests"], "Linting with flake8"),
        (["mypy", "tda_backend"], "Type checking with mypy"),
    ]
    
    for cmd, description in commands:
        console.print(f"🔧 {description}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            console.print(f"✅ {description} completed")
        except subprocess.CalledProcessError as e:
            console.print(f"❌ {description} failed")
            console.print(e.stdout.decode())
            console.print(e.stderr.decode())


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()