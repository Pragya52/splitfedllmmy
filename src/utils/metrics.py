import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingMetrics:
    """Track and visualize training metrics"""
    
    def __init__(self):
        self.client_losses = []
        self.server_losses = []
        self.round_metrics = []
    
    def update_round_metrics(self, round_result: Dict):
        """Update metrics for a federated round"""
        self.round_metrics.append(round_result)
        
        if 'client_losses' in round_result:
            self.client_losses.extend(round_result['client_losses'])
        
        if 'server_losses' in round_result:
            self.server_losses.append(round_result['server_losses'])
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training loss curves"""
        plt.figure(figsize=(12, 4))
        
        # Plot client losses
        plt.subplot(1, 2, 1)
        rounds = [r['round'] for r in self.round_metrics]
        avg_losses = [r['avg_client_loss'] for r in self.round_metrics]
        
        plt.plot(rounds, avg_losses, 'b-', label='Average Client Loss')
        plt.xlabel('Federated Round')
        plt.ylabel('Loss')
        plt.title('Client Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot server losses
        plt.subplot(1, 2, 2)
        if self.server_losses:
            plt.plot(rounds[:len(self.server_losses)], self.server_losses, 'r-', label='Server Loss')
            plt.xlabel('Federated Round')
            plt.ylabel('Loss')
            plt.title('Server Training Loss')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def get_summary(self) -> Dict:
        """Get training summary statistics"""
        if not self.round_metrics:
            return {}
        
        avg_losses = [r['avg_client_loss'] for r in self.round_metrics]
        
        return {
            'total_rounds': len(self.round_metrics),
            'final_loss': avg_losses[-1] if avg_losses else 0.0,
            'best_loss': min(avg_losses) if avg_losses else 0.0,
            'loss_improvement': avg_losses[0] - avg_losses[-1] if len(avg_losses) > 1 else 0.0,
            'convergence_rate': self._calculate_convergence_rate(avg_losses)
        }
    
    def _calculate_convergence_rate(self, losses: List[float]) -> float:
        """Calculate convergence rate"""
        if len(losses) < 2:
            return 0.0
        
        # Simple convergence rate calculation
        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
