import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TrainingMetrics:
    """Enhanced training metrics with individual client tracking"""
    
    def __init__(self):
        self.round_results = []
        self.client_losses_by_round = []  # Track individual client losses
        
    def update_round_metrics(self, round_result: Dict, mode_prefix: str = ""):
        """Update metrics for a training round"""
        # Add mode prefix for comparison experiments
        if mode_prefix:
            prefixed_result = {f"{mode_prefix}_{k}": v for k, v in round_result.items()}
            prefixed_result['mode'] = mode_prefix
            prefixed_result.update(round_result)  # Keep original keys too
            self.round_results.append(prefixed_result)
        else:
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
        if any(loss != 0 for loss in server_losses):
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


class QuantizedTrainingMetrics(TrainingMetrics):
    """Enhanced training metrics with quantization tracking support"""
    
    def __init__(self, track_quantization: bool = True):
        super().__init__()
        self.track_quantization = track_quantization
        self.quantization_mse_by_round = []
        self.client_quantization_stats = []
        
    def update_round_metrics(self, round_result: Dict, mode_prefix: str = ""):
        """Update metrics for a training round with quantization support"""
        super().update_round_metrics(round_result, mode_prefix)
        
        # Track quantization metrics if enabled
        if self.track_quantization:
            if 'server_quantization_mse' in round_result:
                self.quantization_mse_by_round.append(round_result['server_quantization_mse'])
            
            if 'client_quantization_stats' in round_result:
                self.client_quantization_stats.append(round_result['client_quantization_stats'])
    
    def update_quantization_metrics(self, server_mse: float, client_stats: List[Dict]):
        """Update quantization-specific metrics"""
        if self.track_quantization:
            self.quantization_mse_by_round.append(server_mse)
            self.client_quantization_stats.append(client_stats)
    
    def plot_quantized_training_curves(self, save_path: str = None, show_individual_clients: bool = True):
        """Plot training curves with quantization metrics"""
        if not self.round_results:
            print("No training data to plot")
            return
        
        # Create figure with 3 subplots for quantized training
        fig = plt.figure(figsize=(18, 12))
        
        # Extract data
        rounds = [r['round'] for r in self.round_results]
        avg_client_losses = [r['avg_client_loss'] for r in self.round_results]
        server_losses = [r.get('server_losses', 0) for r in self.round_results]
        
        # Plot 1: Average losses (top left)
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(rounds, avg_client_losses, 'b-o', label='Avg Client Loss', linewidth=2, markersize=6)
        if any(loss != 0 for loss in server_losses):
            ax1.plot(rounds, server_losses, 'r-s', label='Server Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Round')
        ax1.set_ylabel('Loss')
        ax1.set_title('Average Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual client losses (top right)
        ax2 = plt.subplot(2, 2, 2)
        if show_individual_clients and self.client_losses_by_round:
            num_clients = len(self.client_losses_by_round[0]) if self.client_losses_by_round else 0
            
            if num_clients > 0:
                colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
                markers = ['o', 's', '^', 'D', 'v', 'p']
                
                for client_id in range(num_clients):
                    client_losses = []
                    for round_losses in self.client_losses_by_round:
                        if client_id < len(round_losses):
                            client_losses.append(round_losses[client_id])
                        else:
                            client_losses.append(None)
                    
                    valid_rounds = [r for r, loss in zip(rounds, client_losses) if loss is not None]
                    valid_losses = [loss for loss in client_losses if loss is not None]
                    
                    if valid_losses:
                        color = colors[client_id % len(colors)]
                        marker = markers[client_id % len(markers)]
                        
                        ax2.plot(valid_rounds, valid_losses, 
                                color=color, marker=marker, label=f'Client {client_id}', 
                                linewidth=2, markersize=5, alpha=0.8)
                
                ax2.plot(rounds, avg_client_losses, 'k--', 
                        label='Average', linewidth=2, alpha=0.7)
                
                ax2.set_xlabel('Federated Round')
                ax2.set_ylabel('Loss')
                ax2.set_title(f'Individual Client Losses ({num_clients} clients)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Quantization MSE (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        if self.track_quantization and self.quantization_mse_by_round:
            ax3.plot(rounds[:len(self.quantization_mse_by_round)], self.quantization_mse_by_round, 
                    'g-^', label='Server Quantization MSE', linewidth=2, markersize=6)
            ax3.set_xlabel('Federated Round')
            ax3.set_ylabel('Quantization MSE')
            ax3.set_title('Quantization Error Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add statistics
            avg_mse = np.mean(self.quantization_mse_by_round)
            ax3.axhline(y=avg_mse, color='red', linestyle='--', alpha=0.7, 
                       label=f'Avg MSE: {avg_mse:.6f}')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No quantization data available', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Quantization scales evolution (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        if self.track_quantization and self.client_quantization_stats:
            # Extract scale evolution for first client as example
            client_to_server_scales = []
            server_to_client_scales = []
            
            for round_stats in self.client_quantization_stats:
                if round_stats and len(round_stats) > 0:
                    first_client_stats = round_stats[0]
                    if first_client_stats.get('quantization_enabled', False):
                        client_to_server_scales.append(first_client_stats.get('client_to_server_scale', 0))
                        server_to_client_scales.append(first_client_stats.get('server_to_client_scale', 0))
            
            if client_to_server_scales and server_to_client_scales:
                scale_rounds = list(range(1, len(client_to_server_scales) + 1))
                ax4.plot(scale_rounds, client_to_server_scales, 'b-o', 
                        label='Client→Server Scale', linewidth=2, markersize=5)
                ax4.plot(scale_rounds, server_to_client_scales, 'r-s', 
                        label='Server→Client Scale', linewidth=2, markersize=5)
                ax4.set_xlabel('Federated Round')
                ax4.set_ylabel('Quantization Scale')
                ax4.set_title('Quantization Scales Evolution (Client 0)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No quantization scale data available', 
                        ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'Quantization tracking disabled', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Quantized training plots saved to {save_path}")
        
        plt.show()
    
    def plot_training_curves(self, save_path: str = None, show_individual_clients: bool = True):
        """Override to use quantized version if quantization is enabled"""
        if self.track_quantization:
            self.plot_quantized_training_curves(save_path, show_individual_clients)
        else:
            super().plot_training_curves(save_path, show_individual_clients)
    
    def get_quantization_summary(self) -> Dict:
        """Get quantization-specific summary statistics"""
        if not self.track_quantization or not self.quantization_mse_by_round:
            return {'quantization_enabled': False}
        
        summary = {
            'quantization_enabled': True,
            'avg_quantization_mse': np.mean(self.quantization_mse_by_round),
            'min_quantization_mse': np.min(self.quantization_mse_by_round),
            'max_quantization_mse': np.max(self.quantization_mse_by_round),
            'quantization_mse_std': np.std(self.quantization_mse_by_round),
            'quantization_mse_trend': 'improving' if len(self.quantization_mse_by_round) > 1 and 
                                     self.quantization_mse_by_round[-1] < self.quantization_mse_by_round[0] else 'stable'
        }
        
        # Add client quantization statistics
        if self.client_quantization_stats:
            final_round_stats = self.client_quantization_stats[-1]
            if final_round_stats:
                summary['num_quantized_clients'] = len([s for s in final_round_stats 
                                                      if s.get('quantization_enabled', False)])
                
                # Average scales across clients
                client_to_server_scales = [s.get('client_to_server_scale', 0) for s in final_round_stats 
                                         if s.get('quantization_enabled', False)]
                server_to_client_scales = [s.get('server_to_client_scale', 0) for s in final_round_stats 
                                         if s.get('quantization_enabled', False)]
                
                if client_to_server_scales:
                    summary['avg_client_to_server_scale'] = np.mean(client_to_server_scales)
                    summary['avg_server_to_client_scale'] = np.mean(server_to_client_scales)
        
        return summary
    
    def print_quantized_summary(self):
        """Print detailed summary including quantization metrics"""
        # Print standard summary first
        self.print_detailed_summary()
        
        # Add quantization summary
        quant_summary = self.get_quantization_summary()
        
        if quant_summary.get('quantization_enabled', False):
            print("\n" + "="*50)
            print("🔢 QUANTIZATION PERFORMANCE SUMMARY")
            print("="*50)
            print(f"📊 Average Quantization MSE: {quant_summary.get('avg_quantization_mse', 0):.6f}")
            print(f"📈 MSE Range: [{quant_summary.get('min_quantization_mse', 0):.6f}, {quant_summary.get('max_quantization_mse', 0):.6f}]")
            print(f"📏 MSE Std Dev: {quant_summary.get('quantization_mse_std', 0):.6f}")
            print(f"📉 MSE Trend: {quant_summary.get('quantization_mse_trend', 'unknown').capitalize()}")
            
            if 'num_quantized_clients' in quant_summary:
                print(f"👥 Quantized Clients: {quant_summary['num_quantized_clients']}")
                
            if 'avg_client_to_server_scale' in quant_summary:
                print(f"⚖️  Avg Client→Server Scale: {quant_summary['avg_client_to_server_scale']:.6f}")
                print(f"⚖️  Avg Server→Client Scale: {quant_summary['avg_server_to_client_scale']:.6f}")
            
            # Calculate compression ratio estimate
            compression_ratio = 32 / 4  # FP32 to INT4
            print(f"🗜️  Estimated Compression Ratio: {compression_ratio:.1f}x (FP32→INT4)")
            
            print("="*50)
        else:
            print("\n📊 Quantization was not enabled or no quantization data available")
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary including quantization metrics"""
        summary = super().get_summary()
        
        # Add quantization summary if enabled
        if self.track_quantization:
            quant_summary = self.get_quantization_summary()
            summary.update(quant_summary)
        
        return summary
