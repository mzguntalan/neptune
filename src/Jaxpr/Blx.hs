{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Jaxpr.Blx where

import Data.List (findIndex, intercalate, nub)
import Data.Map.Strict qualified as Map

-- Blx Before Jax, Above Lax
-- This is a slightly modified mirror of Lax from Jax to Accomodate the Trace Approach
-- focusing on Tracing

data TensorType = Tf32 | Tf16 deriving (Eq)

instance Show TensorType where
    show Tf32 = "f32"
    show Tf16 = "f16"

type Axis = Int -- in future might be literal

type AxisSize = Int -- in future might be literal

type Shape = [AxisSize]

data VarType = Tvar | Tlit

data BlxTensor = LaxTensor TensorType [AxisSize] String VarType

instance Show BlxTensor where
    show (LaxTensor t s n _) = n ++ ":" ++ show t ++ show s

tensor :: TensorType -> [Int] -> String -> VarType -> BlxTensor
tensor = LaxTensor

rank :: BlxTensor -> Int
rank = length . shape

shape :: BlxTensor -> [AxisSize]
shape (LaxTensor _ s _ _) = s

tensorType :: BlxTensor -> TensorType
tensorType (LaxTensor t _ _ _) = t

sameType :: BlxTensor -> BlxTensor -> Bool
sameType t1 t2 = tensorType t1 == tensorType t2

sameShape :: BlxTensor -> BlxTensor -> Bool
sameShape t1 t2 = shape t1 == shape t2

shapeAtAxis :: BlxTensor -> Axis -> AxisSize
shapeAtAxis t n = shape t !! n

tensorName :: BlxTensor -> String -- identifier
tensorName (LaxTensor _ _ n _) = n

renameTensor :: BlxTensor -> String -> BlxTensor
renameTensor (LaxTensor t s _ z) newName = LaxTensor t s newName z

tensorCopy :: BlxTensor -> BlxTensor
tensorCopy t = renameTensor t (tensorName t ++ ".copy")

renameTensorsWithSeedName :: [BlxTensor] -> String -> [BlxTensor]
renameTensorsWithSeedName ts seedName = renamedTensors
  where
    newNames = map (((seedName ++ ".") ++) . show) [1, 2 .. (length ts)]
    renamedTensors = zipWith renameTensor ts newNames

tensorVarType :: BlxTensor -> VarType
tensorVarType (LaxTensor _ _ _ vt) = vt

shapeSingleConcat :: Shape -> Shape -> Axis -> Shape
shapeSingleConcat (a : as) (b : bs) axis = f (a : as) (b : bs) axis 0
  where
    f :: Shape -> Shape -> Axis -> Axis -> Shape
    f (c : cs) (d : ds) target cur
        | target == cur =
            if cs == ds
                then c + d : cs
                else error "shape mismatch"
        | target > cur =
            if c == d
                then c : f cs ds target (cur + 1)
                else error "shape mismatch"
        | otherwise = error "Not enough axes"
    f _ _ _ _ = error "Should not happen"
shapeSingleConcat _ _ _ = error "Shape should be non empty list"

shapeConcat :: [Shape] -> Axis -> Shape
shapeConcat (shap : shapes) axis = foldl (\x y -> shapeSingleConcat x y axis) shap shapes
shapeConcat [] _ = error "no shapes to concat"

data Parameter = Parameter String String

instance Show Parameter where
    show (Parameter n v) = n ++ "=" ++ v

paramList :: [Parameter] -> String
paramList ps = "[" ++ intercalate "," (map show ps) ++ "]"

data LaxPrimitive = Abs | Add | Concatenate {concatenateDimension :: Int} | Var | Lit

instance Show LaxPrimitive where
    show = primRepresentation

primNumInput :: LaxPrimitive -> Int
primNumInput Abs = 1
primNumInput Add = 2
primNumInput (Concatenate _) = -1 -- variable number
primNumInput Var = 1
primNumInput Lit = 1

primNumOutput :: LaxPrimitive -> Int
primNumOutput Abs = 1
primNumOutput Add = 2
primNumOutput (Concatenate _) = -1
primNumOutput Var = 1
primNumOutput Lit = 1

primRepresentation :: LaxPrimitive -> String
primRepresentation Abs = "abs"
primRepresentation Add = "add"
primRepresentation (Concatenate{concatenateDimension = d}) = "concatenate" ++ paramList [Parameter "dimension" (show d)]
primRepresentation Var = "var"
primRepresentation Lit = "lit"

allEq :: (Eq a) => [a] -> Bool
allEq [] = True
allEq [_] = True
allEq [a, b] = a == b
allEq (a1 : (a2 : others)) = a1 == a2 && allEq others

primSimulateApply :: LaxPrimitive -> [BlxTensor] -> [BlxTensor]
primSimulateApply Add [a, b]
    | sameShape a b && sameType a b = [LaxTensor (tensorType a) (shape a) "" Tvar] -- might be Tlit or Tvar IDK yet
primSimulateApply Abs [a] = [a]
primSimulateApply Concatenate{concatenateDimension = d} (t : otherTensors)
    | allEq ranks && allEq targetAxes && allEq types = [LaxTensor commonType resultShape "" Tvar]
  where
    ts = t : otherTensors
    ranks = map rank ts
    targetAxes = map (`shapeAtAxis` d) ts
    types = map tensorType ts
    LaxTensor commonType _ _ _ = t
    resultShape = shapeConcat (map shape ts) d -- TODO: this is wrong, right a shape math lib maybe
primSimulateApply Var [t] = [t]
primSimulateApply Lit [t] = [t]
primSimulateApply _ _ = error "Either not implemented; or you made a mistake or I made a mistake"

type EqInput = BlxTensor

type EqOutput = BlxTensor

type EqPrimitive = LaxPrimitive

data Equation = Equation EqPrimitive [EqInput] [EqOutput]

instance Show Equation where
    show (Equation prim inputs outputs) = showOutputs ++ " = " ++ show prim ++ " " ++ showInputs
      where
        showInputs = unwords (map tensorName inputs)
        showOutputs = unwords (map show outputs)

equation :: EqPrimitive -> [BlxTensor] -> [BlxTensor] -> Equation
equation = Equation

eqPrimitive :: Equation -> LaxPrimitive
eqPrimitive (Equation prim _ _) = prim

eqInputs :: Equation -> [BlxTensor]
eqInputs (Equation _ inputs _) = inputs

eqOutputs :: Equation -> [BlxTensor]
eqOutputs (Equation _ _ outputs) = outputs

eqRenameWithSeed :: Equation -> String -> Equation
eqRenameWithSeed (Equation prim inputs outputs) seedName = Equation prim renamedInputs renamedOutputs
  where
    inputSeedName = seedName ++ ".in."
    outputSeedName = seedName ++ ".out."

    inputNames = map ((inputSeedName ++) . show) [1, 2 .. (length inputs)]
    outputNames = map ((outputSeedName ++) . show) [1, 2 .. (length outputs)]

    renamedInputs :: [BlxTensor]
    renamedInputs = zipWith renameTensor inputs inputNames
    renamedOutputs :: [BlxTensor]
    renamedOutputs = zipWith renameTensor outputs outputNames

data Trace = Trace [Equation] String

instance Show Trace where
    show (Trace eqs n) = "Trace {" ++ n ++ "} [ \n\t" ++ intercalate "\n\t" (map show eqs) ++ "\n]"

traceName :: Trace -> String
traceName (Trace _ n) = n

currentTraceOutputs :: Trace -> [BlxTensor]
currentTraceOutputs (Trace [] _) = []
currentTraceOutputs (Trace (Equation _ _ outputs : _) _) = outputs

traceEquations :: Trace -> [Equation]
traceEquations (Trace eqs _) = eqs

traceJoinEquations :: [Trace] -> [Equation]
traceJoinEquations = concatMap traceEquations . reverse

var :: BlxTensor -> Trace
var t = Trace eqs (tensorName t)
  where
    eq = Equation Var [t] (primSimulateApply Var [t])
    renamedEq = eqRenameWithSeed eq (tensorName t ++ ".1")
    eqs = [renamedEq]

unvar :: Trace -> BlxTensor
unvar (Trace [Equation Var [_] [t]] n) = renameTensor t n
unvar _ = error "You can only unvar a var"

lit :: BlxTensor -> Trace
lit t = Trace eqs (tensorName t)
  where
    eq = Equation Lit [t] (primSimulateApply Var [t])
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

lconcatenate :: [Trace] -> Axis -> Trace
lconcatenate traces axis = Trace (newEquation : equations) newTraceName
  where
    newTraceName = intercalate "" (map traceName traces)
    equations = traceJoinEquations traces
    newEquation = Equation prim inputs outTensors
    inputs = map (head . currentTraceOutputs) traces
    prim = Concatenate{concatenateDimension = axis}
    outTensors = map (`renameTensor` (newTraceName ++ "." ++ show (length equations + 1))) (primSimulateApply prim inputs)

type JaxConst = BlxTensor

type JaxInputVariable = BlxTensor

type JaxOutput = BlxTensor

data JaxExpression = JaxExpression [JaxConst] [JaxInputVariable] [Equation] [JaxOutput]

isVarEquation :: Equation -> Bool
isVarEquation (Equation Var _ _) = True
isVarEquation _ = False

isLitEquation :: Equation -> Bool
isLitEquation (Equation Lit _ _) = True
isLitEquation _ = False

compileTrace :: Trace -> JaxExpression
compileTrace tr = JaxExpression consts inVars eqs outVars
  where
    allTraceEqs = reverse . traceEquations $ tr
    eqs = filter isJaxExpressionEquation allTraceEqs
    isJaxExpressionEquation :: Equation -> Bool
    isJaxExpressionEquation x = not (isLitEquation x) && not (isVarEquation x)

    litEqs = filter isLitEquation allTraceEqs
    varEqs = filter isVarEquation allTraceEqs

    consts = map (head . eqOutputs) litEqs
    inVars = map (head . eqOutputs) varEqs

    outVars = currentTraceOutputs tr

compilePrettyTrace :: Trace -> JaxExpression
compilePrettyTrace = prettifyJaxpr . compileTrace

symbolsForNaming :: String
symbolsForNaming = "abcdefghijklmnopqrstuvwxyz"

newVarName :: String -> String
newVarName [char]
    | char == 'z' = "aa"
    | otherwise = [symbolsForNaming !! iNext]
  where
    i = case findIndex (== char) symbolsForNaming of
        Just a -> a
        Nothing -> error "You're doing things wrong. Recheck the naming pipeline"
    iNext = i + 1
newVarName (x : xs) = newVarName [x] ++ xs
newVarName [] = "a"

-- this needs fixing
varNameFromInt :: Int -> String
varNameFromInt x
    | 0 <= x && x < length symbolsForNaming = [symbolsForNaming !! x]
    | x >= length symbolsForNaming = "a" ++ varNameFromInt (x - length symbolsForNaming)
    | otherwise = error "Shouldn't happen"

-- newVarName

eqNumVars :: Equation -> Int
eqNumVars eq = primNumInput prim + primNumOutput prim where prim = eqPrimitive eq

eqAllVars :: Equation -> [BlxTensor]
eqAllVars eq = eqInputs eq ++ eqOutputs eq

renameEquationUsingMap :: Equation -> Map.Map String String -> Equation
renameEquationUsingMap (Equation prim inputs outputs) varmap = Equation prim renamedInputs renamedOutputs
  where
    f :: BlxTensor -> BlxTensor
    f t = renameTensor t newName
      where
        newName = case Map.lookup (tensorName t) varmap of
            Just a -> a
            Nothing -> error "You made a mistake with generating the correspondences of oldnames to new names"

    renamedInputs = map f inputs
    renamedOutputs = map f outputs

prettifyTrace :: Trace -> Trace
prettifyTrace tr = Trace newEquations (traceName tr)
  where
    newEquations = map (`renameEquationUsingMap` lookupOfVarNames) eqs
    eqs = traceEquations tr
    allVarsInAllEquations = concatMap eqAllVars eqs
    allCurrentVarNames :: [String]
    allCurrentVarNames = nub (map tensorName allVarsInAllEquations)
    upperBoundNumVars = sum (map eqNumVars eqs)
    varnamePool = map varNameFromInt [0, 1 .. (upperBoundNumVars - 1)]
    lookupOfVarNames :: Map.Map String String
    lookupOfVarNames = Map.fromList (zip allCurrentVarNames varnamePool)

renameTensorUsingMap :: BlxTensor -> Map.Map String String -> BlxTensor
renameTensorUsingMap t m = case Map.lookup (tensorName t) m of
    Just newName -> renameTensor t newName
    Nothing -> error "Name not found in lookup"

prettifyJaxpr :: JaxExpression -> JaxExpression
prettifyJaxpr (JaxExpression consts inVars eqs outs) = JaxExpression renamedConsts renamedInVars renamedEqs renamedOuts
  where
    allVars :: [BlxTensor]
    allVars = consts ++ inVars ++ concatMap eqAllVars eqs ++ outs
    allVarNames = nub $ map tensorName allVars
    varnamePool = map varNameFromInt [0, 1 .. (length allVars - 1)]
    lookupOfVarNames :: Map.Map String String
    lookupOfVarNames = Map.fromList (zip allVarNames varnamePool)

    renamedConsts = map (`renameTensorUsingMap` lookupOfVarNames) consts
    renamedInVars = map (`renameTensorUsingMap` lookupOfVarNames) inVars
    renamedEqs = map (`renameEquationUsingMap` lookupOfVarNames) eqs
    renamedOuts = map (`renameTensorUsingMap` lookupOfVarNames) outs

instance Show JaxExpression where
    show (JaxExpression consts inVars eqs outs) = "{ lambda " ++ constShow ++ " ; " ++ inVarsShow ++ ". let\n\t" ++ equationsShow ++ "\n in " ++ outVarsShow ++ " }"
      where
        constShow = unwords (map show consts)
        inVarsShow = unwords (map show inVars)
        equationsShow = intercalate "\n\t" (map show eqs)
        outVarsShow = "(" ++ intercalate "," (map tensorName outs) ++ ",)"

-- test function
-- The resulting Trace of a function corresponds to Jaxpr
testFunction :: Trace -> Trace -> Trace -> Trace -> Trace
testFunction a b c d = lconcatenate [v1, v2] 0
  where
    v1 = ladd a b
    v2 = labs c `ladd` labs d

testFunction2 :: Trace -> Trace -> Trace -> Trace -> Trace
testFunction2 a b c d = ladd s z
  where
    z = lit (tensor Tf32 [2, 2] "z" Tlit) -- how do i scope this to only testFunction2 automatically
    s = ladd a b `ladd` ladd c d
