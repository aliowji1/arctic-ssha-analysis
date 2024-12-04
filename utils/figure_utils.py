import os
from datetime import datetime

class FigureManager:
    """Utility class for managing figure saving across different analysis types"""
    
    def __init__(self, base_dir="figures"):
        self.base_dir = base_dir
        self.subdirs = {
            'cnn': 'cnn_figures',
            'lstm': 'lstm_figures',
            'kmeans': 'kmeans_figures',
            'random_forest': 'random_forest_figures',
            'lstm_analysis': 'lstm_analysis_figures'
        }
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        os.makedirs(self.base_dir, exist_ok=True)
        for subdir in self.subdirs.values():
            os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)
    
    def save_figure(self, fig, name, analysis_type, timestamp=True):
        """Save figure to appropriate directory with optional timestamp"""
        if analysis_type not in self.subdirs:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
        # Add timestamp to filename if requested
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp_str}.png"
        else:
            filename = f"{name}.png"
        
        # Create full path
        save_path = os.path.join(self.base_dir, self.subdirs[analysis_type], filename)
        
        # Save figure
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
        
        return save_path