{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Neptune.Core.Tensor where

import Data.List (intercalate)

-- Everything is a program

data PrimitiveInstruction = PrimitiveInstruction String [(String, String)] deriving (Eq)

instance Show PrimitiveInstruction where
    show (PrimitiveInstruction name parameterTuples) = name ++ "[" ++ intercalate "," (map showPair parameterTuples) ++ "]"
      where
        showPair (paramName, paramValue) = intercalate "=" [paramName, paramValue]

data Program = Immediate PrimitiveInstruction [Program] | ProgramList [Program] deriving (Eq)

instance Show Program where
    show = prettyPrint

var :: String -> Program
var name = createvar
  where
    createvar = createVarInstruction name

createVarInstruction :: String -> Program
createVarInstruction name = Immediate (PrimitiveInstruction "VAR" [("opt", name)]) []

add :: Program -> Program -> Program
add var1 var2 = Immediate (PrimitiveInstruction "ADD" []) [var1, var2]

data NaiveEquation = NaiveEquation String String

instance Show NaiveEquation where
    show (NaiveEquation lhs rhs) = lhs ++ " = " ++ rhs

listify :: Program -> String -> [NaiveEquation]
listify (Immediate prim inputPrograms) seedName = topEq : otherEqs
  where
    topEq = NaiveEquation seedName rhs
    rhs = show prim ++ unwords namesOfInputs
    namesOfInputs = map (((seedName ++ ".") ++) . show) [1, 2 .. length inputPrograms]
    otherEqs = concat $ zipWith listify inputPrograms namesOfInputs
listify (ProgramList programs) seedName = concat $ zipWith listify programs programSeedNames
  where
    programSeedNames = map (((seedName ++ ".") ++) . show) [1, 2 .. length programs]

prettyPrint :: Program -> String
prettyPrint p = intercalate "\n" (map show $ listify p "<>")
