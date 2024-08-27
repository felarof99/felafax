#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import sys
import importlib

# Add the parent directory of the current working directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
def import_local_module(module_path: str):
    module = importlib.import_module(module_path)
    return importlib.reload(module)

try:
    import felafax
    print("felafax package imported successfully")
except ImportError as e:
    print(f"Error importing felafax: {e}")


# In[10]:


setup = import_local_module("trainer_engine.setup")
setup.setup_environment()


# In[12]:


utils = import_local_module("trainer_engine.utils")
jax_utils = import_local_module("trainer_engine.jax_utils")

checkpoint_lib = import_local_module("trainer_engine.checkpoint_lib")
training_pipeline = import_local_module("trainer_engine.trainer_lib")
auto_lib = import_local_module("trainer_engine.auto_lib")

llama_config = import_local_module("llama_config")


# In[3]:





# In[ ]:





# In[4]:


HUGGINGFACE_USERNAME = input("INPUT: Please provide your HUGGINGFACE_USERNAME: ")
HUGGINGFACE_TOKEN = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")


# In[5]:


# Select a supported model from above list to use!
MODEL_NAME = "Meta-Llama-3.1-8B"


# In[7]:


model_path, model, model_config, tokenizer = auto_lib.AutoJAXModelForCausalLM.from_pretrained("llama-3.1-8B-JAX",
                                                                           HUGGINGFACE_TOKEN)

