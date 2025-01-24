{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Jaxpr.Blx.Primitives where

import Data.List (intercalate)
import Debug.Trace (trace)
import Jaxpr.Blx.Tensor

errorSomethingWentWrong :: String
errorSomethingWentWrong = "Something went wrong. Go fix it"

class BlxPrimitive a where
    numInputs :: a -> Int
    numOutputs :: a -> Int
    parameters :: a -> [(String, String)]
    applyPrimitive :: a -> [BlxTensor] -> [BlxTensor]
    symbol :: a -> String

showPrimitive :: (BlxPrimitive a) => a -> String
showPrimitive prim
    | null (parameters prim) = symbol prim
    | otherwise = symbol prim ++ "[" ++ intercalate "," (map (\(x, y) -> x ++ "=" ++ y) (parameters prim)) ++ "]"

data Abs = Abs

instance BlxPrimitive Abs where
    numInputs Abs = 1
    numOutputs Abs = 1
    parameters Abs = []
    applyPrimitive Abs [a] = [a]
    applyPrimitive Abs _ = error errorSomethingWentWrong
    symbol Abs = "abs"

data Add = Add

instance BlxPrimitive Add where
    numInputs Add = 2
    numOutputs Add = 1
    parameters Add = []
    applyPrimitive Add [a, b] = [c] where c = BlxTensor (tensorType a) (tensorShape b) "" Tvar
    applyPrimitive Add _ = error errorSomethingWentWrong
    symbol Add = "add"

data Concatenate
    = Concatenate
        Int -- dimension

instance BlxPrimitive Concatenate where
    numInputs (Concatenate _) = -1
    numOutputs (Concatenate _) = 1
    parameters (Concatenate d) = [("dimension", show d)]
    applyPrimitive (Concatenate d) (t : ts) = [BlxTensor (tensorType t) (shapeConcat (map tensorShape (t : ts)) d) "" (tensorDesignation t)]
    applyPrimitive _ _ = error errorSomethingWentWrong
    symbol (Concatenate _) = "concatenate"

data Var = Var

instance BlxPrimitive Var where
    numInputs Var = 1
    numOutputs Var = 1
    parameters Var = []
    applyPrimitive Var [t] = [t]
    applyPrimitive _ _ = error errorSomethingWentWrong
    symbol Var = "var"

data Lit = Lit

instance BlxPrimitive Lit where
    numInputs Lit = 1
    numOutputs Lit = 1
    parameters Lit = []
    applyPrimitive Lit [t] = [t]
    applyPrimitive _ _ = error errorSomethingWentWrong
    symbol Lit = "lit"
