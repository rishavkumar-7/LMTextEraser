class GPU:
    _instance = None  # Class variable to hold the single instance

    def __new__(cls, gpu_id=None):
        if cls._instance is None:
            # Create the single instance if it doesn't exist
            cls._instance = super(GPU, cls).__new__(cls)
            cls._instance.gpu_id = gpu_id  # Initialize the gpu_id
        else:
            # Update the gpu_id if a new one is provided
            if gpu_id is not None:
                cls._instance.gpu_id = gpu_id
        return cls._instance

    def get_gpu(self):
        return self.gpu_id
