# onnx-util
## Infer Shape
    from onnx_util.onnx_util import *
    
    model = infer_shape(input_path:str, temp_model_path:str)
    onnx.save(model,model_path)
    
## Extract_model
    from onnx_util.onnx_util import *
    
    extract_model(input_path:str,output_path:str,input_names: list[str],output_names: list[str])
    
## Reorder_node
    import onnx
    from onnx_util.onnx_util import *

    model = onnx.load(model_path)
    model = reorder_node(model)
    onnx.save(model,model_path)
