{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Jaxpr.Blx.Primitives where

import Data.List (intercalate)
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
showPrimitive prim = symbol prim ++ "[" ++ intercalate "," (map (\(x, y) -> x ++ "=" ++ y) (parameters prim)) ++ "]"

data Abs = Abs

data Add = Add

data Concatenate
    = Concatenate
        Int -- dimension

instance BlxPrimitive Abs where
    numInputs Abs = 1
    numOutputs Abs = 1
    parameters Abs = []
    applyPrimitive Abs [a] = [a]
    applyPrimitive Abs _ = error errorSomethingWentWrong
    symbol Abs = "abs"

instance BlxPrimitive Add where
    numInputs Add = 1
    numOutputs Add = 1
    parameters Add = []
    applyPrimitive Add [a, b] = [c] where c = BlxTensor (tensorType a) (tensorShape b) "" Tvar
    applyPrimitive Add _ = error errorSomethingWentWrong
    symbol Add = "add"

instance BlxPrimitive Concatenate where
    numInputs (Concatenate _) = -1
    numOutputs (Concatenate _) = 1
    parameters (Concatenate d) = [("dimension", show d)]
    applyPrimitive (Concatenate d) (t : ts) = [BlxTensor (tensorType t) (shapeConcat (map tensorShape (t : ts)) d) "" (tensorDesignation t)]
    applyPrimitive _ _ = error errorSomethingWentWrong
    symbol (Concatenate _) = "concatenate"
