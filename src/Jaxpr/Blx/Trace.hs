module Jaxpr.Blx.Trace where

import Data.List (intercalate)
import Jaxpr.Blx.Primitives
import Jaxpr.Blx.Tensor

data TracePrimitive = forall a. (BlxPrimitive a) => TracePrimitive a

instance BlxPrimitive TracePrimitive where
    numInputs (TracePrimitive p) = numInputs p
    numOutputs (TracePrimitive p) = numOutputs p
    parameters (TracePrimitive p) = parameters p
    applyPrimitive (TracePrimitive p) = applyPrimitive p
    symbol (TracePrimitive p) = symbol p

instance Eq TracePrimitive where
    a == b = show a == show b

instance Show TracePrimitive where show (TracePrimitive a) = showPrimitive a

data TraceEntry = TraceEntry TracePrimitive [BlxTensor] [BlxTensor]

entryPrimitive :: TraceEntry -> TracePrimitive
entryPrimitive (TraceEntry prim _ _) = prim

entryInputs :: TraceEntry -> [BlxTensor]
entryInputs (TraceEntry _ inputs _) = inputs

entryOutputs :: TraceEntry -> [BlxTensor]
entryOutputs (TraceEntry _ _ outputs) = outputs

entryRenameWithSeed :: TraceEntry -> String -> TraceEntry
entryRenameWithSeed (TraceEntry prim inputs outputs) seedName = TraceEntry prim renamedInputs renamedOutputs
  where
    inputSeedName = seedName ++ ".in."
    outputSeedName = seedName ++ ".out."

    inputNames = map ((inputSeedName ++) . show) [1, 2 .. (length inputs)]
    outputNames = map ((outputSeedName ++) . show) [1, 2 .. (length outputs)]

    renamedInputs :: [BlxTensor]
    renamedInputs = zipWith renameTensor inputs inputNames
    renamedOutputs :: [BlxTensor]
    renamedOutputs = zipWith renameTensor outputs outputNames

instance Show TraceEntry where
    show (TraceEntry prim inputs outputs) = outputsShow ++ " = " ++ show prim ++ " " ++ unwords (map tensorName inputs)
      where
        outputsShow = case length outputs of
            0 -> error "NO OUTPUT for primitive! It does nothing!"
            1 -> intercalate "," (map show outputs)
            _ -> "(" ++ intercalate "," (map show outputs) ++ ")"

data BlxTrace = Trace [TraceEntry] String

instance Show BlxTrace where
    show (Trace entries name) = "Trace {" ++ name ++ "} [\n\t" ++ intercalate "\n\t" (map show entries) ++ "\n]"

traceName :: BlxTrace -> String
traceName (Trace _ n) = n

currentTraceOutputs :: BlxTrace -> [BlxTensor]
currentTraceOutputs (Trace [] _) = []
currentTraceOutputs (Trace (lastEntry : _) _) = entryOutputs lastEntry

traceEntries :: BlxTrace -> [TraceEntry]
traceEntries (Trace entries _) = entries

traceJoinEntries :: [BlxTrace] -> [TraceEntry]
traceJoinEntries = concatMap traceEntries . reverse

mkTensorTrace :: TensorType -> [Int] -> String -> Designation -> BlxTrace
mkTensorTrace ttype tshape tname tdesig = Trace entries tname
  where
    -- entries = [entryRenameWithSeed entry (tname ++ ".1")]
    entries = [entry]
    entry = TraceEntry prim [t] (applyPrimitive prim [t])
    prim = TracePrimitive Var
    t = BlxTensor ttype tshape tname tdesig

var :: TensorType -> [Int] -> String -> BlxTrace
var ttype tshape tname = mkTensorTrace ttype tshape tname Tvar

lit :: TensorType -> [Int] -> String -> BlxTrace
lit ttype tshape tname = mkTensorTrace ttype tshape tname Tvar
