#!/usr/bin/env python3
"""Script to run federated server"""

import argparse
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.llama_server import FederatedServer

def main():
    parser = argparse.ArgumentParser(description="SplitFedLLM Server")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Config file")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup
    config = Config.from_yaml(args.config)
    logger = setup_logger(__name__, "logs/server.log")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    
    logger.info(f"Starting server on port {args.port} with device {device}")
    
    # Note: In a real distributed setting, implement actual network communication
    print(f"Server would listen on port {args.port}")
    
    logger.info("Server setup completed")

if __name__ == "__main__":
    main()
