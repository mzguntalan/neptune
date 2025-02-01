module Nepure.Core.Program where

import Data.List (intercalate)

data PrimitiveInstruction a = (Eq a, Show a) => PrimitiveInstruction a [(String, String)]

instance Eq (PrimitiveInstruction a) where
    (PrimitiveInstruction prim1 params1) == (PrimitiveInstruction prim2 params2) = prim1 == prim2 && params1 == params2

instance Show (PrimitiveInstruction a) where
    show (PrimitiveInstruction prim parameterTuples) = show prim ++ "[" ++ intercalate "," (map showPair parameterTuples) ++ "]"
      where
        showPair (paramName, paramValue) = intercalate "=" [paramName, paramValue]

data Program a = Immediate (PrimitiveInstruction a) [Program a] | ProgramList [Program a] deriving (Eq)

instance Show (Program a) where
    show = prettyPrint

data NaiveEquation = NaiveEquation String String

instance Show NaiveEquation where
    show (NaiveEquation lhs rhs) = lhs ++ " = " ++ rhs

listify :: Program a -> String -> [NaiveEquation]
listify (Immediate prim inputPrograms) seedName = topEq : otherEqs
  where
    topEq = NaiveEquation seedName rhs
    rhs = show prim ++ unwords namesOfInputs
    namesOfInputs = map (((seedName ++ ".") ++) . show) [1, 2 .. length inputPrograms]
    otherEqs = concat $ zipWith listify inputPrograms namesOfInputs
listify (ProgramList programs) seedName = concat $ zipWith listify programs programSeedNames
  where
    programSeedNames = map (((seedName ++ ".") ++) . show) [1, 2 .. length programs]

prettyPrint :: Program a -> String
prettyPrint p = intercalate "\n" (map show $ listify p "<>")

var :: String -> Program String
var name = createvar
  where
    createvar = createVarInstruction name

createVarInstruction :: String -> Program String
createVarInstruction name = Immediate (PrimitiveInstruction "VAR" [("opt", name)]) []

add :: Program String -> Program String -> Program String
add var1 var2 = Immediate (PrimitiveInstruction "ADD" []) [var1, var2]
