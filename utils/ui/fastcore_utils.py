"""
Utility module for FastAI compatibility with FastCore.
This provides compatibility layers for fastcore.transform and other modules
needed for FastAI model loading when using newer PyTorch versions.
"""

import sys
import types
import pip
import os
import platform
import torch
import pickle

def setup_path_compatibility():
    """
    Set up compatibility for loading models with PosixPath on Windows systems.
    This patches the unpickler to convert PosixPath to WindowsPath during model loading.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Only needed on Windows
        if platform.system() != 'Windows':
            return True
            
        # Import pathlib components
        from pathlib import Path, PosixPath, WindowsPath
        
        # Create a custom unpickler that handles PosixPath
        class PathFixUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'pathlib' and name == 'PosixPath':
                    # Replace PosixPath with the proper Path for the current OS
                    return WindowsPath
                return super().find_class(module, name)
        
        # Create a patched version of torch.load that uses our custom unpickler
        original_torch_load = torch.load
        def patched_torch_load(f, map_location=None, pickle_module=pickle, **kwargs):
            if pickle_module == pickle:  # Only patch if using the default pickle
                opened_file = False
                try:
                    # If f is a string (file path), open the file
                    if isinstance(f, str):
                        opened_file = True
                        f_obj = open(f, 'rb')
                    else:
                        # Assume it's already a file-like object
                        f_obj = f
                    
                    # Use our custom unpickler on the file object
                    unpickler = PathFixUnpickler(f_obj)
                    if map_location is not None:
                        unpickler.persistent_load = lambda saved_id: map_location(saved_id)
                    
                    # Load the data
                    result = unpickler.load()
                    
                    # Close if we opened it
                    if opened_file:
                        f_obj.close()
                    
                    return result
                except Exception as e:
                    # If our custom approach fails, close file if we opened it
                    if opened_file and 'f_obj' in locals():
                        f_obj.close()
                    
                    # Use the original torch.load with a wrapper that modifies the pickle module
                    class SafeUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if module == 'pathlib' and name == 'PosixPath':
                                return WindowsPath
                            return super().find_class(module, name)
                    
                    safe_pickle = pickle.Unpickler
                    pickle.Unpickler = SafeUnpickler
                    try:
                        result = original_torch_load(f, map_location=map_location, 
                                                    pickle_module=pickle_module, **kwargs)
                        return result
                    finally:
                        # Restore the original Unpickler
                        pickle.Unpickler = safe_pickle
            else:
                # Use the original function if a custom pickle module is provided
                return original_torch_load(f, map_location=map_location, 
                                          pickle_module=pickle_module, **kwargs)
        
        # Replace torch.load with our patched version
        torch.load = patched_torch_load
        print("✓ Path compatibility layer installed for Windows")
        return True
    
    except Exception as e:
        print(f"ERROR setting up path compatibility: {str(e)}")
        return False

def setup_fastcore_compatibility():
    """
    Set up the compatibility layer for fastcore.transform and related modules.
    This ensures FastAI models can be unpickled correctly even when fastcore is not available.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    # First set up path compatibility
    path_compatibility_success = setup_path_compatibility()
    
    if "fastcore.transform" not in sys.modules or "fastcore.dispatch" not in sys.modules:
        try:
            # First ensure fasttransform is installed
            try:
                import fasttransform
            except ImportError:
                print("Installing fasttransform package...")
                pip.main(['install', 'git+https://github.com/AnswerDotAI/fasttransform.git'])
                print("✓ fasttransform installed successfully.")
                import fasttransform
            
            # Create compatibility module for fastcore.transform and populate it
            print("Setting up fastcore.transform compatibility layer...")
            fastcore_transform = types.ModuleType("fastcore.transform")
            
            # Create minimal class definitions for essential Transform classes
            # Only try to import what actually exists, with fallbacks for missing classes
            
            # Start with Pipeline which is the main one needed for model loading
            if hasattr(fasttransform, 'Pipeline'):
                from fasttransform import Pipeline
                fastcore_transform.Pipeline = Pipeline
            else:
                # Create a minimal Pipeline implementation if it doesn't exist
                class Pipeline:
                    def __init__(self, *args, **kwargs): pass
                fastcore_transform.Pipeline = Pipeline
                print("  Created Pipeline placeholder class")
            
            # Try to import other classes with fallbacks
            try:
                from fasttransform import Transform
                fastcore_transform.Transform = Transform
            except (ImportError, AttributeError):
                # Create minimal Transform class if needed
                class Transform:
                    def __init__(self, *args, **kwargs): pass
                fastcore_transform.Transform = Transform
                print("  Created Transform placeholder class")
            
            # Add other required classes with fallbacks
            class_names = ['DisplayedTransform', 'ItemTransform', 'TupleTransform']
            for class_name in class_names:
                try:
                    cls = getattr(fasttransform, class_name)
                    setattr(fastcore_transform, class_name, cls)
                except (ImportError, AttributeError):
                    # Create a minimal placeholder class
                    placeholder = type(class_name, (), {"__init__": lambda self, *args, **kwargs: None})
                    setattr(fastcore_transform, class_name, placeholder)
                    print(f"  Created {class_name} placeholder class")
            
            # For functions, try to import them or create simple fallbacks
            function_names = ['drop_none', 'retain_type', 'get_func']
            for func_name in function_names:
                try:
                    func = getattr(fasttransform, func_name)
                    setattr(fastcore_transform, func_name, func)
                except (ImportError, AttributeError):
                    # Create a simple pass-through function
                    setattr(fastcore_transform, func_name, lambda *args, **kwargs: args[0] if args else None)
                    print(f"  Created {func_name} placeholder function")
            
            # Register the transform module with Python
            sys.modules["fastcore.transform"] = fastcore_transform
            
            # Create and set up fastcore.dispatch compatibility module
            print("Setting up fastcore.dispatch compatibility layer...")
            fastcore_dispatch = types.ModuleType("fastcore.dispatch")
            
            # Add the missing _TypeDict class that's needed for unpickling
            class _TypeDict(dict):
                """Simple implementation of _TypeDict to satisfy the unpickler"""
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                
                def add(self, t, f=None):
                    if f is not None:
                        self[t] = f
                    return self
                
                def __repr__(self):
                    return f"_TypeDict({super().__repr__()})"
            
            # Add _TypeDict to fastcore.dispatch
            fastcore_dispatch._TypeDict = _TypeDict
            print("  Created _TypeDict compatibility class")
            
            # Import TypeDispatch or create a fallback
            try:
                from plum import Function
                # Use Function from plum as a replacement for TypeDispatch
                class TypeDispatch:
                    def __init__(self, *funcs, **kwargs):
                        self.func = Function(funcs[0] if funcs else lambda: None)
                        self.types = _TypeDict()  # Use our _TypeDict implementation
                        self._resolution_cache = {}  # Add a cache for method resolutions
                        for f in funcs[1:]:
                            self.func.register(f)
                    
                    def __call__(self, *args, **kwargs):
                        return self.func(*args, **kwargs)
                    
                    def add(self, f):
                        self.func.register(f)
                        return self
                    
                    # Add other necessary methods for compatibility
                    def returns(self, x):
                        # Simple implementation to avoid complex logic
                        return None
                    
                    def __getitem__(self, key):
                        # Simple implementation to avoid complex logic
                        return lambda x: x
                    
                    def __repr__(self):
                        return str(self.func.methods)
                    
                    # Add the missing method that's causing the error
                    def _resolve_method_with_cache(self, args):
                        # Create a cache key based on argument types
                        arg_types = tuple(type(arg) for arg in args if arg is not None)
                        
                        # Check if we have a cached result
                        if arg_types in self._resolution_cache:
                            return self._resolution_cache[arg_types]
                        
                        # Default implementation: just return a callable function and None for return type
                        # This will make the method always use the first registered function or a simple passthrough
                        if hasattr(self.func, 'methods') and self.func.methods:
                            method = next(iter(self.func.methods), lambda *a, **k: args[0] if args else None)
                        else:
                            method = lambda *a, **k: args[0] if args else None
                            
                        result = (method, None)
                        
                        # Cache the result
                        self._resolution_cache[arg_types] = result
                        return result
                
                fastcore_dispatch.TypeDispatch = TypeDispatch
                print("  Created TypeDispatch compatibility class using plum.Function")
                
                # Add typedispatch decorator
                def typedispatch(f=None):
                    # Simple typedispatch decorator using plum's dispatch
                    from plum import dispatch
                    if f is None: return dispatch
                    return dispatch(f)
                    
                fastcore_dispatch.typedispatch = typedispatch
                print("  Created typedispatch compatibility function")
                
            except (ImportError, AttributeError) as e:
                print(f"  Error setting up TypeDispatch: {str(e)}")
                # Create minimal TypeDispatch class as fallback
                class TypeDispatch:
                    def __init__(self, *args, **kwargs): 
                        self._resolution_cache = {}  # Add a cache for method resolutions
                        
                    def __call__(self, *args, **kwargs): 
                        return args[0] if args else None
                        
                    def add(self, f): return self
                    def returns(self, x): return None
                    def __getitem__(self, key): return lambda x: x
                    def __repr__(self): return "TypeDispatch(placeholder)"
                    
                    # Add the missing method that's causing the error
                    def _resolve_method_with_cache(self, args):
                        # Simple implementation: return a passthrough function and None for return type
                        return (lambda *a, **k: a[0] if a else None, None)
                
                fastcore_dispatch.TypeDispatch = TypeDispatch
                fastcore_dispatch.typedispatch = lambda f=None: (lambda g: g) if f is None else f
                print("  Created TypeDispatch fallback placeholder class")
            
            # Register the dispatch module with Python
            sys.modules["fastcore.dispatch"] = fastcore_dispatch
                
            # Also create fastcore.basics if needed
            if "fastcore.basics" not in sys.modules:
                fastcore_basics = types.ModuleType("fastcore.basics")
                sys.modules["fastcore.basics"] = fastcore_basics
                
            print("✓ Compatibility layer installed successfully")
            return True
        except Exception as e:
            print(f"ERROR setting up compatibility layer: {str(e)}")
            print("Models may fail to load.")
            return False
    
    # Already set up
    return True
