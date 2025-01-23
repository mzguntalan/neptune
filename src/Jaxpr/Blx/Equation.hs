module Jaxpr.Blx.Equation where

import Jaxpr.Blx.Tensor

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
