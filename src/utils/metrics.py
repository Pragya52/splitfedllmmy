import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

class TrainingMetrics:
    """Enhanced training metrics with individual client tracking"""
    
    def __init__(self):
        self.round_results = []
        self.client_losses_by_round = []  # Track individual client losses
        
    def update_round_metrics(self, round_result: Dict):
        """Update metrics for a training round"""
        self.round_results.append(round_result)
        
        # Store individual client losses for this round
        if 'client_losses' in round_result:
            self.client_losses_by_round.append(round_result['client_losses'])
    
    def plot_training_curves(self, save_path: str = None, show_individual_clients: bool = True):
        """Plot training curves with individual client loss tracking"""
        if not self.round_results:
            print("No training data to plot")
            return
        
        # Extract data
        rounds = [r['round'] for r in self.round_results]
        avg_client_losses = [r['avg_client_loss'] for r in self.round_results]
        server_losses = [r.get('server_losses', 0) for r in self.round_results]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average losses
        ax1.plot(rounds, avg_client_losses, 'b-o', label='Avg Client Loss', linewidth=2, markersize=6)
        ax1.plot(rounds, server_losses, 'r-s', label='Server Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Round')
        ax1.set_ylabel('Loss')
        ax1.set_title('Average Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual client losses
        if show_individual_clients and self.client_losses_by_round:
            num_clients = len(self.client_losses_by_round[0]) if self.client_losses_by_round else 0
            
            if num_clients > 0:
                # Fixed color and marker combinations
                colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
                markers = ['o', 's', '^', 'D', 'v', 'p']
                
                for client_id in range(num_clients):
                    client_losses = []
                    for round_losses in self.client_losses_by_round:
                        if client_id < len(round_losses):
                            client_losses.append(round_losses[client_id])
                        else:
                            client_losses.append(None)
                    
                    # Filter out None values
                    valid_rounds = [r for r, loss in zip(rounds, client_losses) if loss is not None]
                    valid_losses = [loss for loss in client_losses if loss is not None]
                    
                    if valid_losses:
                        color = colors[client_id % len(colors)]
                        marker = markers[client_id % len(markers)]
                        
                        ax2.plot(valid_rounds, valid_losses, 
                                color=color, marker=marker, label=f'Client {client_id}', 
                                linewidth=2, markersize=5, alpha=0.8)
                
                # Also plot average for comparison
                ax2.plot(rounds, avg_client_losses, 'k--', 
                        label='Average', linewidth=2, alpha=0.7)
                
                ax2.set_xlabel('Federated Round')
                ax2.set_ylabel('Loss')
                ax2.set_title(f'Individual Client Losses ({num_clients} clients)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No individual client data available', 
                        ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()
        
    def get_summary(self) -> Dict:
        """Get training summary statistics"""
        if not self.round_results:
            return {}
        
        # Calculate convergence metrics
        initial_loss = self.round_results[0]['avg_client_loss']
        final_loss = self.round_results[-1]['avg_client_loss']
        improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
        
        # Client diversity metrics
        client_std_devs = []
        for round_losses in self.client_losses_by_round:
            if len(round_losses) > 1:
                client_std_devs.append(np.std(round_losses))
        
        avg_client_diversity = np.mean(client_std_devs) if client_std_devs else 0
        
        summary = {
            'total_rounds': len(self.round_results),
            'initial_avg_loss': initial_loss,
            'final_avg_loss': final_loss,
            'improvement_percent': improvement,
            'avg_client_diversity': avg_client_diversity,
            'convergence_rate': improvement / len(self.round_results) if len(self.round_results) > 0 else 0
        }
        
        # Individual client final losses
        if self.client_losses_by_round:
            final_client_losses = self.client_losses_by_round[-1]
            for i, loss in enumerate(final_client_losses):
                summary[f'client_{i}_final_loss'] = loss
        
        return summary
    
    def print_detailed_summary(self):
        """Print detailed training summary"""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("🎯 FEDERATED TRAINING SUMMARY")
        print("="*50)
        print(f"📊 Total Rounds: {summary.get('total_rounds', 'N/A')}")
        print(f"🚀 Initial Avg Loss: {summary.get('initial_avg_loss', 0):.4f}")
        print(f"🎯 Final Avg Loss: {summary.get('final_avg_loss', 0):.4f}")
        print(f"📈 Improvement: {summary.get('improvement_percent', 0):.1f}%")
        print(f"⚡ Convergence Rate: {summary.get('convergence_rate', 0):.3f}% per round")
        print(f"🔄 Client Diversity: {summary.get('avg_client_diversity', 0):.4f}")
        
        # Individual client performance
        print("\n📱 Individual Client Performance:")
        client_keys = [k for k in summary.keys() if k.startswith('client_') and k.endswith('_final_loss')]
        for key in sorted(client_keys):
            client_id = key.split('_')[1]
            loss = summary[key]
            print(f"   Client {client_id}: {loss:.4f}")
        
        print("="*50)
