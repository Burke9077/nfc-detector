"""
Utility module for FastAI compatibility with FastCore.
This provides compatibility layers for fastcore.transform and other modules
needed for FastAI model loading when using newer PyTorch versions.
"""

import sys
import types
import pip

def setup_fastcore_compatibility():
    """
    Set up the compatibility layer for fastcore.transform and related modules.
    This ensures FastAI models can be unpickled correctly even when fastcore is not available.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
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
            
            # Import TypeDispatch or create a fallback
            try:
                from plum import Function
                # Use Function from plum as a replacement for TypeDispatch
                class TypeDispatch:
                    def __init__(self, *funcs, **kwargs):
                        self.func = Function(funcs[0] if funcs else lambda: None)
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
                        return lambda x
                    
                    def __repr__(self):
                        return str(self.func.methods)
                
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
                    def __init__(self, *args, **kwargs): pass
                    def __call__(self, *args, **kwargs): return args[0] if args else None
                    def add(self, f): return self
                    def returns(self, x): return None
                    def __getitem__(self, key): return lambda x: x
                    def __repr__(self): return "TypeDispatch(placeholder)"
                    
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
