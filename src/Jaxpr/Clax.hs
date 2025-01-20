{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Clax where

-- focusing on Tracing

type VarName = String
type FuncName = String

data Vartype = Vi | Vf deriving (Show)
data Variable = Variable VarName Vartype deriving (Show)
type Output = Variable
type Input = Variable
data Function = Function FuncName deriving (Show)
data Equation = Equation [Output] Function [Input] deriving (Show)

type CurrentValue = Variable
data Tracer = Tracer CurrentValue [Equation] deriving (Show)

var :: Variable -> Tracer
var x = Tracer x []

id :: Tracer -> Tracer
id (Tracer curval eqs) = Tracer curval (Equation [curval] (Function "id") [curval] : eqs)

add :: Tracer -> Tracer -> Tracer
add (Tracer x xEqs) (Tracer y yEqs) = Tracer z (Equation [z] (Function "add") [x, y] : eqs)
  where
    eqs = xEqs ++ yEqs
    z = Variable "out" Vf
