module Jaxpr.Blx.Lax where

import Data.List (intercalate)
import Jaxpr.Blx.Primitives
import Jaxpr.Blx.Tensor
import Jaxpr.Blx.Trace

-- the following are lax mirrors

-- attempt at generalizing
laxApply :: TracePrimitive -> [BlxTrace] -> BlxTrace
laxApply prim inputTraces
    | numInputs prim == length inputTraces || numInputs prim == (-1) = Trace (newEntry : combinedTraceEntries) newTraceName
    | otherwise = error "Wrong Number of Inputs"
  where
    newEntry = TraceEntry prim inputs outTensors
    newTraceName = intercalate "" (map traceName inputTraces)
    combinedTraceEntries = traceJoinEntries inputTraces
    inputs = map (head . currentTraceOutputs) inputTraces
    outTensors = map (`renameTensor` (newTraceName ++ "." ++ show (length combinedTraceEntries + 1))) (applyPrimitive prim inputs)

labs :: BlxTrace -> BlxTrace
labs tr = case currentTraceOutputs tr of
    [a] -> Trace (newEquation : traceEntries tr) (traceName tr)
      where
        newEquation :: TraceEntry
        prim = TracePrimitive Abs
        newEquation = TraceEntry prim [a] outTensors
        outTensors = map (`renameTensor` (traceName tr ++ "." ++ show ((length . traceEntries $ tr) + 1))) (applyPrimitive prim [a])
    _ -> error "Too many inputs"

ladd :: BlxTrace -> BlxTrace -> BlxTrace
ladd trX trY = laxApply (TracePrimitive Add) [trX, trY]

lconcatenate :: [BlxTrace] -> Axis -> BlxTrace
lconcatenate traces axis = laxApply (TracePrimitive (Concatenate axis)) traces

-- -- limited to rank 0 tensors for now
lbroadcastInDim :: BlxTrace -> Shape -> BlxTrace
lbroadcastInDim tr targetShape = laxApply (TracePrimitive (BroadcastInDim [] targetShape)) [tr]

-- -- limited to rank 0 tensors for now
-- lbroadcastInDim :: BlxTrace -> Shape -> BlxTrace
-- lbroadcastInDim tr targetShape = Trace (newEquation : equations) newTraceName
--   where
--     p = BroadcastInDim{broadcastInDimDimensions = [], broadcastInDimShape = targetShape}
--     onlyTensorInTraceOutput = case currentTraceOutputs tr of
--         [t] -> t
--         _ -> error ("There should only be 1 and only 1 tensor in the trace " ++ traceName tr)
--     outputs = map (`renameTensor` (newTraceName ++ "." ++ show (length equations + 1))) (primApply p [onlyTensorInTraceOutput])
--     newEquation = BlxEquation p [onlyTensorInTraceOutput] outputs
--     newTraceName = traceName tr
--     equations = traceEquations tr
