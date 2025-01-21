{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE StandaloneDeriving #-}

module Jaxpr.Clax where

import Data.List (intercalate)

-- focusing on Tracing

data NoneArrayType = Cint | Cstring

data NonArray = NonArray NoneArrayType String

data ArrayType = Af32 | Ai32

data Array a = Array [a] ArrayType String

data ArrayTemplate a = ArrayTemplate [a] ArrayType

data NoneArrayTemplate = NoneArrayTemplate NoneArrayType

class Value a where
    valueName :: a -> String

class ValueTemplate a where
    toValue :: (Value b) => a -> String -> b

instance Value (Array a) where
    valueName (Array _ _ n) = n

instance Value NonArray where
    valueName (NonArray _ n) = n

instance ValueTemplate (ArrayTemplate a) where
    -- toValue :: ArrayTemplate a -> String -> Array a
    toValue (ArrayTemplate v t) = Array v t

instance ValueTemplate NoneArrayTemplate where
    toValue :: NoneArrayTemplate -> String -> NonArray
    toValue (NoneArrayTemplate t) = NonArray t

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

numInputOfPrimitiveSymbol :: PrimitiveSymbol -> Int
numInputOfPrimitiveSymbol Abs = 1
numInputOfPrimitiveSymbol Add = 2
numInputOfPrimitiveSymbol Concatenate = 1
numInputOfPrimitiveSymbol Id = 1

numOutputOfPrimitiveSymbol :: PrimitiveSymbol -> Int
numOutputOfPrimitiveSymbol Abs = 1
numOutputOfPrimitiveSymbol Add = 2
numOutputOfPrimitiveSymbol Concatenate = 1
numOutputOfPrimitiveSymbol Id = 1

instance Show Primitive where
    show (Primitive sym []) = show sym
    show (Primitive sym params) = show sym ++ "[" ++ intercalate "," (map show params) ++ "]"

type Input a = a

type Output a = a

data Equation where
    Equation :: (Value a, Value b) => Primitive -> [Input a] -> [Output b] -> Equation

-- createNArrayTemplates :: Int -> [Array a]

createNArrays :: Int -> String -> [Array a]
createNArrays n seedName = zipWith (curry toValue) (createNArrayTemplates n) (createNnames n seedName)

createNArrayTemplates :: Int -> [ArrayTemplate a]
createNArrayTemplates n = map (\_ -> ArrayTemplate [] Af32) [1, 2 .. n]

createNnames :: Int -> String -> [String]
createNnames n seedName = map f [1, 2 .. n]
  where
    f :: Int -> String
    f i = seedName ++ "." ++ show i

applyPrimitive :: (Value a, Value b) => Primitive -> [a] -> Equation
applyPrimitive prim inputs
    | length inputs == numInputOfPrimitiveSymbol sym = Equation prim inputs outputs
    | otherwise = error "Shouldn't happen"
  where
    (Primitive sym _) = prim
    outputs = createNArrayValues (numOutputOfPrimitiveSymbol sym) s
    s = head inputs

type CurrentValues a = a

data Trace where
    Trace :: (Value a) => [a] -> [Equation] -> Trace

variadic :: Primitive -> [Trace] -> Trace
variadic prim ts = Trace outs (eq : eqs)
  where
    eqs :: [Equation]
    eqs = concatMap extractEqs ts
    extractEqs :: Trace -> [Equation]
    extractEqs (Trace _ es) = es
    outs = applyPrimitive prim
