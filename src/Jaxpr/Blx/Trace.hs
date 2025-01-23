module Jaxpr.Blx.Trace where

import Jaxpr.Blx.Equation
import Jaxpr.Blx.Primitives
import Jaxpr.Blx.Tensor

data BlxTrace = Trace [BlxEquation] String

instance Show BlxTrace where
    show (Trace eqs n) = "Trace {" ++ n ++ "} [ \n\t" ++ intercalate "\n\t" (map show eqs) ++ "\n]"

traceName :: BlxTrace -> String
traceName (Trace _ n) = n

currentTraceOutputs :: BlxTrace -> [BlxTensor]
currentTraceOutputs (Trace [] _) = []
currentTraceOutputs (Trace (BlxEquation _ _ outputs : _) _) = outputs

traceEquations :: BlxTrace -> [BlxEquation]
traceEquations (Trace eqs _) = eqs

traceJoinEquations :: [BlxTrace] -> [BlxEquation]
traceJoinEquations = concatMap traceEquations . reverse

var :: TensorType -> [Int] -> String -> BlxTrace
var ttype tshape tname = Trace eqs (tensorName t)
  where
    t = BlxTensor ttype tshape tname Tvar
    eq = BlxEquation Var [t] (primApply Var [t])
    renamedEq = eqRenameWithSeed eq (tensorName t ++ ".1")
    eqs = [renamedEq]

unvar :: BlxTrace -> BlxTensor
unvar (Trace [BlxEquation Var [_] [BlxTensor tt ts tn Tvar]] n) = renameTensor t n where t = BlxTensor tt ts tn Tvar
unvar _ = error "You can only unvar a var"

lit :: TensorType -> [Int] -> String -> BlxTrace
lit ttype tshape tname = Trace eqs (tensorName t)
  where
    t = BlxTensor ttype tshape tname Tlit
    eq = BlxEquation Lit [t] (primApply Lit [t])
    renamedEq = eqRenameWithSeed eq (tensorName t ++ ".1")
    eqs = [renamedEq]

unlit :: BlxTrace -> BlxTensor
unlit (Trace [BlxEquation Lit [_] [BlxTensor tt ts tn Tlit]] n) = renameTensor t n where t = BlxTensor tt ts tn Tlit
unlit _ = error "You can only unlit a lit"

mkTrace :: [BlxEquation] -> String -> BlxTrace
mkTrace eqs s = Trace renamedEqs s
  where
    renamedEqs = zipWith eqRenameWithSeed eqs seedNames
    seedNames = map (((s ++ ".") ++) . show) (reverse [1, 2 .. (length eqs)])
