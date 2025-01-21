{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Clax where

import Data.List (intercalate)

-- focusing on Tracing

data NonArray = Cint | Cfloat | Cstring

data ArrayType = Af32 | Ai32

data Array a = Array [a] ArrayType

class Value a

instance Value Array

instance Value NonArray

data PrimitiveSymbol = Abs | Add | Concatenate | Id

instance Show PrimitiveSymbol where
    show Abs = "abs"
    show Add = "add"
    show Concatenate = "concatenate"
    show Id = "unchecked:id"

type ParamName = String

type ParamValue = String

data Parameter = Parameter ParamName ParamValue

instance Show Parameter where
    show (Parameter name val) = name ++ "=" ++ val

data Primitive = Primitive PrimitiveSymbol [Parameter]

instance Show Primitive where
    show (Primitive sym []) = show sym
    show (Primitive sym params) = show sym ++ "[" ++ intercalate "," (map show params) ++ "]"

type Input a = a

type Output a = a

data Equation where
    Equation :: (Value a) => Primitive -> [Input a] -> [Output a] -> Equation

type CurrentValues a = a

data Trace where
    Trace :: (Value a) => [a] -> [Equation] -> Trace

variadic :: [Trace] -> Trace
variadic ts = Trace outs (eq : eqs)
  where
    eqs :: [Equation]
    eqs = concatMap extractEqs ts
    extractEqs :: Trace -> [Equation]
    extractEqs (Trace _ es) = es
