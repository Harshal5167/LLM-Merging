import torch
from merging.Merges import Merges
# from peft import get_peft_model, set_peft_model_state_dict

class flan_small(Merges):
    def __init__(self, name):
        super().__init__(name)

        '''
        These values are meant to be modified by the user.
        '''
        # Give a list of models to load for the merge 
        self.list_models = [("prudhvirazz/google-flan-t5-small-modified", "8a183c2bb257fbc0cec7ac05d4f5fe0b166d4049"),
                            ("google/flan-t5-small", "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab")]
        
        # Hyperparameters 
        self.base_model_name = "google/flan-t5-small"
        self.base_model_revision_id = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
        self.is_peft = False  # Changed to False as we're not using LoRA models
        self.max_seq_len = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model. 
        self.architecture = "encoder_decoder"

        '''
        These are variables used later in the code and not intended to be set, but feel free to adapt to your use case.  
        '''
        # Loaded models and configs 
        self.loaded_models = {}
        self.loaded_configs = {}

        # Merged model parameters
        self.merged_model = {}

    def slerp(self, low, high, t):
        """Spherical Linear Interpolation."""
        low_norm = low / torch.norm(low)
        high_norm = high / torch.norm(high)
        omega = torch.acos((low_norm * high_norm).sum())
        so = torch.sin(omega)
        return torch.sin((1.0 - t) * omega) / so * low + torch.sin(t * omega) / so * high

    # Implement merge function 
    def merge(self):
        '''
        1) Load HuggingFace checkpoints and configs 
        '''
        super()._load_huggingface_models_and_configs()

        '''
        2) Merge checkpoints using SLERP
        '''
        t = 0.5  # Interpolation factor, 0.5 for equal weight

        # Get individual models 
        model1, model2 = list(self.loaded_models.values())

        # Get all the parameters names (assumes both models have the same parameters)
        all_parameter_names = model1.keys()

        for parameter_name in all_parameter_names:
            param1 = model1[parameter_name]
            param2 = model2[parameter_name]
            
            # Apply SLERP
            merged_parameter = self.slerp(param1, param2, t)
            
            self.merged_model[parameter_name] = merged_parameter

        '''
        3) Load base model and tokenizer 
        '''
        self._load_base_model()
        self._load_tokenizer()

        '''
        4) Load merged model into base model 
        '''
        self.base_model.load_state_dict(self.merged_model)

        # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass. 
        self.base_model.eval()

        return self.base_model