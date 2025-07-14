#!/usr/bin/env python3
"""Script to run individual federated client"""

import argparse
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.llama_client import FederatedClient

def main():
    parser = argparse.ArgumentParser(description="FL-LLaMA Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--server_address", type=str, default="localhost", help="Server address")
    parser.add_argument("--server_port", type=int, default=8080, help="Server port")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup
    config = Config.from_yaml(args.config)
    logger = setup_logger(__name__, f"logs/client_{args.client_id}.log")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    
    logger.info(f"Starting client {args.client_id} on device {device}")
    
    # Initialize client
    # Note: In a real distributed setting, you would implement actual network communication
    # For now, this is a placeholder for the client-side implementation
    
    logger.info(f"Client {args.client_id} setup completed")

if __name__ == "__main__":
    main()
