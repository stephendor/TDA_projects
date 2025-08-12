#!/bin/bash
export GEMINI_API_KEY="AIzaSyCNZKgWVAwcU8ZhBDWU0KNR_iYkIwt94eg"
exec /home/stephen-dorman/dev/TDA_projects/.venv/bin/uvx --from git+https://github.com/BeehiveInnovations/zen-mcp-server.git zen-mcp-server "$@"