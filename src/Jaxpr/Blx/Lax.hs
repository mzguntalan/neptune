module Jaxpr.Blx.Lax where

import Jaxpr.Blx.Equation

-- the following are lax mirrors
labs :: BlxTrace -> BlxTrace
labs tr = case currentTraceOutputs tr of
    [a] -> Trace (newEquation : traceEquations tr) (traceName tr)
      where
        newEquation :: BlxEquation
        newEquation = BlxEquation Abs [a] outTensors
        outTensors = map (`renameTensor` (traceName tr ++ "." ++ show ((length . traceEquations $ tr) + 1))) (primApply Abs [a])
    _ -> error "Too many inputs"

ladd :: BlxTrace -> BlxTrace -> BlxTrace
ladd trX trY = Trace (newEquation : equations) newTraceName
  where
    equations = traceJoinEquations [trX, trY]
    newTraceName = traceName trX ++ traceName trY
    newEquation = BlxEquation Add [x, y] outTensors
    outTensors = map (`renameTensor` (newTraceName ++ "." ++ show (length equations + 1))) (primApply Add [x, y])
    [x] = currentTraceOutputs trX
    [y] = currentTraceOutputs trY

lconcatenate :: [BlxTrace] -> Axis -> BlxTrace
lconcatenate traces axis = Trace (newEquation : equations) newTraceName
  where
    newTraceName = intercalate "" (map traceName traces)
    equations = traceJoinEquations traces
    newEquation = BlxEquation prim inputs outTensors
    inputs = map (head . currentTraceOutputs) traces
    prim = Concatenate{concatenateDimension = axis}
    outTensors = map (`renameTensor` (newTraceName ++ "." ++ show (length equations + 1))) (primApply prim inputs)

-- limited to rank 0 tensors for now
lbroadcastInDim :: BlxTrace -> Shape -> BlxTrace
lbroadcastInDim tr targetShape = Trace (newEquation : equations) newTraceName
  where
    p = BroadcastInDim{broadcastInDimDimensions = [], broadcastInDimShape = targetShape}
    onlyTensorInTraceOutput = case currentTraceOutputs tr of
        [t] -> t
        _ -> error ("There should only be 1 and only 1 tensor in the trace " ++ traceName tr)
    outputs = map (`renameTensor` (newTraceName ++ "." ++ show (length equations + 1))) (primApply p [onlyTensorInTraceOutput])
    newEquation = BlxEquation p [onlyTensorInTraceOutput] outputs
    newTraceName = traceName tr
    equations = traceEquations tr
