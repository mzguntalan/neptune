{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Jaxpr.Clax where

import Data.List (intercalate)

-- focusing on Tracing

data TensorType = Tf32 | Tf16 deriving (Eq)

instance Show TensorType where
    show Tf32 = "f32"
    show Tf16 = "f16"

type Axis = Int -- in future might be literal

type AxisSize = Int -- in future might be literal

data LaxTensor = LaxTensor TensorType [AxisSize] String

instance Show LaxTensor where
    show (LaxTensor t s n) = n ++ ":" ++ show t ++ show s

tensor :: TensorType -> [Int] -> String -> LaxTensor
tensor = LaxTensor

rank :: LaxTensor -> Int
rank = length . shape

shape :: LaxTensor -> [AxisSize]
shape (LaxTensor _ s _) = s

tensorType :: LaxTensor -> TensorType
tensorType (LaxTensor t _ _) = t

sameType :: LaxTensor -> LaxTensor -> Bool
sameType t1 t2 = tensorType t1 == tensorType t2

sameShape :: LaxTensor -> LaxTensor -> Bool
sameShape t1 t2 = shape t1 == shape t2

shapeAtAxis :: LaxTensor -> Axis -> AxisSize
shapeAtAxis t n = shape t !! n

tensorName :: LaxTensor -> String -- identifier
tensorName (LaxTensor _ _ n) = n

renameTensor :: LaxTensor -> String -> LaxTensor
renameTensor (LaxTensor t s _) = LaxTensor t s

data Parameter = Parameter String String

instance Show Parameter where
    show (Parameter n v) = n ++ "=" ++ v

paramList :: [Parameter] -> String
paramList ps = "[" ++ intercalate "," (map show ps) ++ "]"

data LaxPrimitive = Abs | Add | Concatenate {concatenateDimensions :: Int} | Var

instance Show LaxPrimitive where
    show = primRepresentation

primNumInput :: LaxPrimitive -> Int
primNumInput Abs = 1
primNumInput Add = 2
primNumInput (Concatenate _) = -1 -- variable number
primNumInput Var = 1

primNumOutput :: LaxPrimitive -> Int
primNumOutput Abs = 1
primNumOutput Add = 2
primNumOutput (Concatenate _) = -1
primNumOutput Var = 1

primRepresentation :: LaxPrimitive -> String
primRepresentation Abs = "abs"
primRepresentation Add = "add"
primRepresentation (Concatenate{concatenateDimensions = d}) = "concatenate" ++ paramList [Parameter "dimension" (show d)]
primRepresentation Var = "var"

allEq :: (Eq a) => [a] -> Bool
allEq [] = True
allEq [_] = True
allEq [a, b] = a == b
allEq (a1 : (a2 : others)) = a1 == a2 && allEq others

primSimulateApply :: LaxPrimitive -> [LaxTensor] -> [LaxTensor]
primSimulateApply Add [a, b]
    | sameShape a b && sameType a b = [LaxTensor (tensorType a) (shape a) ""]
primSimulateApply Abs [a] = [a]
primSimulateApply Concatenate{concatenateDimensions = d} (t : otherTensors)
    | allEq ranks && allEq targetAxes && allEq types = [LaxTensor commonType resultShape ""]
  where
    ts = t : otherTensors
    ranks = map rank ts
    targetAxes = map (`shapeAtAxis` d) ts
    types = map tensorType ts
    LaxTensor commonType _ _ = t
    resultShape = shape t -- TODO: this is wrong, right a shape math lib maybe
primSimulateApply Var [t] = [t]
primSimulateApply _ _ = error "Either not implemented; or you made a mistake or I made a mistake"

type EqInput = LaxTensor

type EqOutput = LaxTensor

type EqPrimitive = LaxPrimitive

data Equation = Equation EqPrimitive [EqInput] [EqOutput]

instance Show Equation where
    show (Equation prim inputs outputs) = showOutputs ++ "=" ++ show prim ++ " " ++ showInputs
      where
        showInputs = unwords (map show inputs)
        showOutputs = unwords (map show outputs)

equation :: EqPrimitive -> [LaxTensor] -> [LaxTensor] -> Equation
equation = Equation

eqPrimitive :: Equation -> LaxPrimitive
eqPrimitive (Equation prim _ _) = prim

eqInputs :: Equation -> [LaxTensor]
eqInputs (Equation _ inputs _) = inputs

eqOutputs :: Equation -> [LaxTensor]
eqOutputs (Equation _ _ outputs) = outputs

eqRenameWithSeed :: Equation -> String -> Equation
eqRenameWithSeed (Equation prim inputs outputs) seedName = Equation prim renamedInputs renamedOutputs
  where
    inputSeedName = seedName ++ ".in."
    outputSeedName = seedName ++ ".out."

    inputNames = map ((inputSeedName ++) . show) [1, 2 .. (length inputs)]
    outputNames = map ((outputSeedName ++) . show) [1, 2 .. (length outputs)]

    renamedInputs :: [LaxTensor]
    renamedInputs = zipWith renameTensor inputs inputNames
    renamedOutputs :: [LaxTensor]
    renamedOutputs = zipWith renameTensor outputs outputNames

data Trace = Trace [Equation] String

instance Show Trace where
    show (Trace eqs n) = "Trace {" ++ n ++ "} [ \n\t" ++ intercalate "\n\t" (map show eqs) ++ "\n]"

traceName :: Trace -> String
traceName (Trace _ n) = n

currentTraceOutputs :: Trace -> [LaxTensor]
currentTraceOutputs (Trace [] _) = []
currentTraceOutputs (Trace (Equation _ _ outputs : _) _) = outputs

traceEquations :: Trace -> [Equation]
traceEquations (Trace eqs _) = eqs

traceJoinEquations :: [Trace] -> [Equation]
traceJoinEquations = concatMap traceEquations

var :: LaxTensor -> Trace
var t = Trace eqs (tensorName t)
  where
    eq = Equation Var [t] (primSimulateApply Var [t])
    renamedEq = eqRenameWithSeed eq (tensorName t ++ ".1")
    eqs = [renamedEq]

mkTrace :: [Equation] -> String -> Trace
mkTrace eqs s = Trace renamedEqs s
  where
    renamedEqs = zipWith eqRenameWithSeed eqs seedNames
    seedNames = map (((s ++ ".") ++) . show) (reverse [1, 2 .. (length eqs)])

-- the following are lax mirrors
labs :: Trace -> Trace
labs tr = case currentTraceOutputs tr of
    [a] -> Trace (newEquation : traceEquations tr) (traceName tr)
      where
        newEquation :: Equation
        newEquation = Equation Abs [a] outTensors
        outTensors = map (`renameTensor` (traceName tr ++ "." ++ show ((length . traceEquations $ tr) + 1))) (primSimulateApply Abs [a])
    _ -> error "Too many inputs"

ladd :: Trace -> Trace -> Trace
ladd trX trY = Trace (newEquation : equations) newTraceName
  where
    equations = traceJoinEquations [trX, trY]
    newTraceName = traceName trX ++ traceName trY
    newEquation = Equation Add [x, y] outTensors
    outTensors = map (`renameTensor` (newTraceName ++ "." ++ show (length equations + 1))) (primSimulateApply Add [x, y])
    [x] = currentTraceOutputs trX
    [y] = currentTraceOutputs trY

-- trace should reflect the input of the last primitive applied OR last function applied
