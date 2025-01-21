{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Clax where

import Data.List (intercalate)

-- focusing on Tracing

data TensorType = Tf32 | Tf16

type Axis = Int -- in future might be literal

type AxisSize = Int -- in future might be literal

class Tensor a where
    rank :: a -> Int
    shape :: a -> [AxisSize]
    sameType :: a -> a -> Bool
    tensorType :: a -> TensorType
    sameShape :: a -> a -> Bool
    shapeAtAxis :: a -> Int -> Axis -> AxisSize
    name :: a -> String -- identifier

class Template a where
    realizeAsTensor :: (Tensor b) => a -> String -> b
    dummyTensor :: (Tensor b) => a -> b
    dummyTensor template = realizeAsTensor template (templateName template)
    templateName :: a -> String
    templateName _ = "dummy"

class Primitive a where
    pNumInput :: a -> Int
    pNumOutput :: a -> Int
    pSimulateApply :: (Tensor b, Tensor c) => a -> [b] -> [c]

class Equation a where
    eqPrim :: (Primitive p) => a -> p
    eqInputs :: (Tensor t) => a -> [t]
    eqOutputs :: (Tensor t) => a -> [t]
    renameWithSeed :: a -> String -> a

class Trace a where
    currentOutputs :: (Tensor b) => a -> [b]
    trEquations :: (Equation e) => a -> [e]
    trJoinEquations :: (Equation e) => [a] -> [e]
    trJoinEquations = concatMap trEquations
