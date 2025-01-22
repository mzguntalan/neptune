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

data Designation = Tvar | Tlit

data BlxTensor = BlxTensor TensorType [AxisSize] String Designation

instance Show BlxTensor where
    show (BlxTensor t s n _) = n ++ ":" ++ show t ++ show s

tensor :: TensorType -> [Int] -> String -> Designation -> BlxTensor
tensor = BlxTensor

tensorRank :: BlxTensor -> Int
tensorRank = length . tensorShape

tensorShape :: BlxTensor -> [AxisSize]
tensorShape (BlxTensor _ s _ _) = s

tensorType :: BlxTensor -> TensorType
tensorType (BlxTensor t _ _ _) = t

sameType :: BlxTensor -> BlxTensor -> Bool
sameType t1 t2 = tensorType t1 == tensorType t2

sameShape :: BlxTensor -> BlxTensor -> Bool
sameShape t1 t2 = tensorShape t1 == tensorShape t2

shapeAtAxis :: BlxTensor -> Axis -> AxisSize
shapeAtAxis t n = tensorShape t !! n

tensorName :: BlxTensor -> String -- identifier
tensorName (BlxTensor _ _ n _) = n

renameTensor :: BlxTensor -> String -> BlxTensor
renameTensor (BlxTensor t s _ z) newName = BlxTensor t s newName z

tensorCopy :: BlxTensor -> BlxTensor
tensorCopy t = renameTensor t (tensorName t ++ ".copy")

renameTensorsWithSeedName :: [BlxTensor] -> String -> [BlxTensor]
renameTensorsWithSeedName ts seedName = renamedTensors
  where
    newNames = map (((seedName ++ ".") ++) . show) [1, 2 .. (length ts)]
    renamedTensors = zipWith renameTensor ts newNames

tensorDesignation :: BlxTensor -> Designation
tensorDesignation (BlxTensor _ _ _ vt) = vt

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

data BlxPrimParameter = BlxPrimParameter String String

instance Show BlxPrimParameter where
    show (BlxPrimParameter n v) = n ++ "=" ++ v

paramList :: [BlxPrimParameter] -> String
paramList ps = "[" ++ intercalate "," (map show ps) ++ "]"

data BlxPrimitive = Abs | Add | Concatenate {concatenateDimension :: Int} | Var | Lit

instance Show BlxPrimitive where
    show = primRepresentation

primNumInput :: BlxPrimitive -> Int
primNumInput Abs = 1
primNumInput Add = 2
primNumInput (Concatenate _) = -1 -- variable number
primNumInput Var = 1
primNumInput Lit = 1

primNumOutput :: BlxPrimitive -> Int
primNumOutput Abs = 1
primNumOutput Add = 2
primNumOutput (Concatenate _) = -1
primNumOutput Var = 1
primNumOutput Lit = 1

primRepresentation :: BlxPrimitive -> String
primRepresentation Abs = "abs"
primRepresentation Add = "add"
primRepresentation (Concatenate{concatenateDimension = d}) = "concatenate" ++ paramList [BlxPrimParameter "dimension" (show d)]
primRepresentation Var = "var"
primRepresentation Lit = "lit"

allEq :: (Eq a) => [a] -> Bool
allEq [] = True
allEq [_] = True
allEq [a, b] = a == b
allEq (a1 : (a2 : others)) = a1 == a2 && allEq others

primApply :: BlxPrimitive -> [BlxTensor] -> [BlxTensor]
primApply Add [a, b]
    | sameShape a b && sameType a b = [BlxTensor (tensorType a) (tensorShape a) "" Tvar] -- might be Tlit or Tvar IDK yet
primApply Abs [a] = [a]
primApply Concatenate{concatenateDimension = d} (t : otherTensors)
    | allEq ranks && allEq targetAxes && allEq types = [BlxTensor commonType resultShape "" Tvar]
  where
    ts = t : otherTensors
    ranks = map tensorRank ts
    targetAxes = map (`shapeAtAxis` d) ts
    types = map tensorType ts
    BlxTensor commonType _ _ _ = t
    resultShape = shapeConcat (map tensorShape ts) d -- TODO: this is wrong, right a shape math lib maybe
primApply Var [t] = [t]
primApply Lit [t] = [t]
primApply _ _ = error "Either not implemented; or you made a mistake or I made a mistake"

type EqInput = BlxTensor

type EqOutput = BlxTensor

type EqPrimitive = BlxPrimitive

data BlxEquation = BlxEquation EqPrimitive [EqInput] [EqOutput]

instance Show BlxEquation where
    show (BlxEquation prim inputs outputs) = showOutputs ++ " = " ++ show prim ++ " " ++ showInputs
      where
        showInputs = unwords (map tensorName inputs)
        showOutputs = unwords (map show outputs)

equation :: EqPrimitive -> [BlxTensor] -> [BlxTensor] -> BlxEquation
equation = BlxEquation

eqPrimitive :: BlxEquation -> BlxPrimitive
eqPrimitive (BlxEquation prim _ _) = prim

eqInputs :: BlxEquation -> [BlxTensor]
eqInputs (BlxEquation _ inputs _) = inputs

eqOutputs :: BlxEquation -> [BlxTensor]
eqOutputs (BlxEquation _ _ outputs) = outputs

eqRenameWithSeed :: BlxEquation -> String -> BlxEquation
eqRenameWithSeed (BlxEquation prim inputs outputs) seedName = BlxEquation prim renamedInputs renamedOutputs
  where
    inputSeedName = seedName ++ ".in."
    outputSeedName = seedName ++ ".out."

    inputNames = map ((inputSeedName ++) . show) [1, 2 .. (length inputs)]
    outputNames = map ((outputSeedName ++) . show) [1, 2 .. (length outputs)]

    renamedInputs :: [BlxTensor]
    renamedInputs = zipWith renameTensor inputs inputNames
    renamedOutputs :: [BlxTensor]
    renamedOutputs = zipWith renameTensor outputs outputNames

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

type JaxConst = BlxTensor

type JaxInputVariable = BlxTensor

type JaxOutput = BlxTensor

data JaxExpression = JaxExpression [JaxConst] [JaxInputVariable] [BlxEquation] [JaxOutput]

isVarEquation :: BlxEquation -> Bool
isVarEquation (BlxEquation Var _ _) = True
isVarEquation _ = False

isLitEquation :: BlxEquation -> Bool
isLitEquation (BlxEquation Lit _ _) = True
isLitEquation _ = False

traceToJaxExpression :: BlxTrace -> JaxExpression
traceToJaxExpression tr = JaxExpression consts inVars eqs outVars
  where
    allTraceEqs = reverse . traceEquations $ tr
    eqs = filter isJaxExpressionEquation allTraceEqs
    isJaxExpressionEquation :: BlxEquation -> Bool
    isJaxExpressionEquation x = not (isLitEquation x) && not (isVarEquation x)

    litEqs = filter isLitEquation allTraceEqs
    varEqs = filter isVarEquation allTraceEqs

    consts = map (head . eqOutputs) litEqs
    inVars = map (head . eqOutputs) varEqs

    outVars = currentTraceOutputs tr

compileTrace :: BlxTrace -> JaxExpression
compileTrace = prettifyJaxpr . traceToJaxExpression

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

eqNumVars :: BlxEquation -> Int
eqNumVars eq = primNumInput prim + primNumOutput prim where prim = eqPrimitive eq

eqAllVars :: BlxEquation -> [BlxTensor]
eqAllVars eq = eqInputs eq ++ eqOutputs eq

renameEquationUsingMap :: BlxEquation -> Map.Map String String -> BlxEquation
renameEquationUsingMap (BlxEquation prim inputs outputs) varmap = BlxEquation prim renamedInputs renamedOutputs
  where
    f :: BlxTensor -> BlxTensor
    f t = renameTensor t newName
      where
        newName = case Map.lookup (tensorName t) varmap of
            Just a -> a
            Nothing -> error "You made a mistake with generating the correspondences of oldnames to new names"

    renamedInputs = map f inputs
    renamedOutputs = map f outputs

prettifyTrace :: BlxTrace -> BlxTrace
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
testFunction :: BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace
testFunction a b c d = lconcatenate [v1, v2] 0
  where
    v1 = ladd a b
    v2 = labs c `ladd` labs d

testFunction2 :: BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace -> BlxTrace
testFunction2 a b c d = ladd s z
  where
    z = lit (tensor Tf32 [2, 2] "z" Tlit) -- how do i scope this to only testFunction2 automatically
    s = ladd a b `ladd` ladd c d
