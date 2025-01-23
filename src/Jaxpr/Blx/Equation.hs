{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module Jaxpr.Blx.Equation where

import Jaxpr.Blx.Primitives
import Jaxpr.Blx.Tensor


class SimpleEquation (a b c) where 
    

-- type EqInput = BlxTensor

-- type EqOutput = BlxTensor

-- type EqPrimitive = BlxPrimitive

-- data BlxEquation a = (BlxPrimitive a) => BlxEquation a [EqInput] [EqOutput]

-- instance Show (BlxEquation a) where
--     show (BlxEquation prim inputs outputs) = showOutputs ++ " = " ++ showPrimitive prim ++ " " ++ showInputs
--       where
--         showInputs = unwords (map tensorName inputs)
--         showOutputs = unwords (map show outputs)

-- equation :: (BlxPrimitive a) => a -> [BlxTensor] -> [BlxTensor] -> BlxEquation a
-- equation = BlxEquation

-- eqPrimitive :: (BlxPrimitive a) => BlxEquation a -> a
-- eqPrimitive (BlxEquation prim _ _) = prim

-- eqInputs :: (BlxPrimitive a) => BlxEquation a -> [BlxTensor]
-- eqInputs (BlxEquation _ inputs _) = inputs

-- eqOutputs :: (BlxPrimitive a) => BlxEquation a -> [BlxTensor]
-- eqOutputs (BlxEquation _ _ outputs) = outputs

-- eqRenameWithSeed :: (BlxPrimitive a) => BlxEquation a -> String -> BlxEquation a
-- eqRenameWithSeed (BlxEquation prim inputs outputs) seedName = BlxEquation prim renamedInputs renamedOutputs
--   where
--     inputSeedName = seedName ++ ".in."
--     outputSeedName = seedName ++ ".out."

--     inputNames = map ((inputSeedName ++) . show) [1, 2 .. (length inputs)]
--     outputNames = map ((outputSeedName ++) . show) [1, 2 .. (length outputs)]

--     renamedInputs :: [BlxTensor]
--     renamedInputs = zipWith renameTensor inputs inputNames
--     renamedOutputs :: [BlxTensor]
--     renamedOutputs = zipWith renameTensor outputs outputNames
