from codecarbon import EmissionsTracker

class CarbonTracker:
    def __init__(self):
        self.tracker = None

    def track(self, model_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Create a new tracker for each function call
                tracker = EmissionsTracker(
                    project_name=model_name,
                    output_dir="emissions",
                    log_level='warning',
                    save_to_file=True,
                    allow_multiple_runs=True
                )
                
                tracker.start()
                result = func(*args, **kwargs)
                emissions = tracker.stop()
                
                if emissions is not None:
                    print(f"\nCarbon emissions from {model_name}: {emissions:.4f} kg CO2eq")
                else:
                    print(f"\nCarbon emissions from {model_name}: Unable to calculate")
                
                return result
            return wrapper
        return decorator