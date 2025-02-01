module Neptune.Core.Program where

import Data.List (intercalate)

data PrimitiveInstruction a = (Eq a, Show a) => PrimitiveInstruction a

instance Eq (PrimitiveInstruction a) where
    (PrimitiveInstruction prim1) == (PrimitiveInstruction prim2) = prim1 == prim2

instance Show (PrimitiveInstruction a) where
    show (PrimitiveInstruction prim) = show prim

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
    rhs = show prim ++ " " ++ unwords namesOfInputs
    namesOfInputs = map (((seedName ++ ".") ++) . show) [1, 2 .. length inputPrograms]
    otherEqs = concat $ zipWith listify inputPrograms namesOfInputs
listify (ProgramList programs) seedName = concat $ zipWith listify programs programSeedNames
  where
    programSeedNames = map (((seedName ++ ".") ++) . show) [1, 2 .. length programs]

prettyPrint :: Program a -> String
prettyPrint p = intercalate "\n" (map show $ listify p "<>")

lastPrimitiveInstructionInProgram :: Program a -> PrimitiveInstruction a
lastPrimitiveInstructionInProgram (Immediate prim _) = prim
lastPrimitiveInstructionInProgram (ProgramList []) = error "empty program"
lastPrimitiveInstructionInProgram (ProgramList (p : _)) = lastPrimitiveInstructionInProgram p
